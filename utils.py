from __future__ import annotations

import os
import ast
import re
from typing import Any
import time

import numpy as np
from pyboy import PyBoy
from PIL import Image
import torch
from trl.extras.profiling import profiling_context


BUTTONS = ["a", "b", "up", "down", "left", "right", "noop"]

ANSWER_RE = re.compile(r"<answer>\s*(\[.*?\])\s*</answer>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are playing Super Mario Land.\n\n"
    "The goal is to progress through levels, collect coins and power-ups when safe, "
    "and ultimately finish the game by rescuing Princess Daisy.\n\n"
    "You can control the game by pressing buttons on the Game Boy.\n\n"
    "Available buttons:\n"
    "- 'a': Jump (used to make Mario jump)\n"
    "- 'b': Run/Shoot (hold to run faster or shoot fireballs if available)\n"
    "- 'up': Climb ladders or vines (if present)\n"
    "- 'down': Crouch or enter pipes (when standing on a pipe)\n"
    "- 'left': Move Mario left\n"
    "- 'right': Move Mario right\n"
    "- 'noop': Do nothing (used to wait for a brief period without performing any action)\n\n"
    "Please analyze the game screen and decide which buttons to press to progress.\n\n"
    "Return your answer as follows:\n"
    "1. Button sequence: a list of buttons to press simultaneously\n"
    "2. Each button should be one of: 'a', 'b', 'up', 'down', 'left', 'right', 'noop'\n\n"
    "First describe what you see on the screen in <perception></perception>. "
    "Then, in <reasoning></reasoning>, break down your reasoning step by step, "
    "justifying each action you consider. Output your final action in "
    "<answer>['button1', 'button2', ...]</answer>.\n\n"
    "The maximum number of buttons you can press simultaneously in one turn is 2."
)

class SuperMarioLandEnv:
    def __init__(
        self,
        rom_path: str,
        *,
        max_turns: int = 256,
        obs_shape: tuple[int, int] = (144, 160),   # (H, W); native GB
        jump_frames: int = 15,
        move_frames: int = 5,
        render_mode: str | None = None,
        w_progress: float = 0.1,
        w_coin: float = 1.0,
        death_penalty: float = -15.0,
        level_clear_bonus: float = 50.0,
        time_penalty: float = -0.01,
    ):
        self.rom_path = rom_path
        self.max_turns = max_turns
        self.obs_h, self.obs_w = obs_shape
        self.jump_frames = jump_frames
        self.move_frames = move_frames
        self.render_mode = render_mode
        self.w_progress = w_progress
        self.w_coin = w_coin
        self.death_penalty = death_penalty
        self.level_clear_bonus = level_clear_bonus
        self.time_penalty = time_penalty

        self.pyboy: PyBoy | None = None
        self.gw = None
        self._turn = 0
        self._prev: dict[str, Any] = {}
        self._last_reward = 0.0
        self._done = False

    def _state(self) -> dict[str, Any]:
        gw = self.gw
        return {
            "turn": self._turn,
            "level_progress": int(gw.level_progress),
            "coins": int(gw.coins),
            "lives_left": int(gw.lives_left),
            "score": int(gw.score),
            "time_left": int(gw.time_left),
            "world": tuple(gw.world),
        }

    def _frame(self) -> np.ndarray:
        img = self.pyboy.screen.image.convert("RGB")
        if img.size != (self.obs_w, self.obs_h):
            img = img.resize((self.obs_w, self.obs_h))
        return np.asarray(img, dtype=np.uint8)

    def _reward(self, cur: dict[str, Any]) -> tuple[float, bool, str | None]:
        prev = self._prev
        d_prog = max(0, cur["level_progress"] - prev["level_progress"])
        reward = self.w_progress * d_prog
        reward += self.w_coin * max(0, cur["coins"] - prev["coins"])
        reward += self.time_penalty
        if cur["lives_left"] < prev["lives_left"] or self.gw.game_over():
            return reward + self.death_penalty, True, "death"
        if cur["world"] != prev["world"]:
            return reward + self.level_clear_bonus, True, "level_clear"
        return float(reward), False, None

    def reset(self) -> np.ndarray:
        if self.pyboy is None:
            window = "SDL2" if self.render_mode == "human" else "null"
            self.pyboy = PyBoy(self.rom_path, window=window)
            self.pyboy.set_emulation_speed(1 if self.render_mode == "human" else 0)
            self.gw = self.pyboy.game_wrapper
            self.gw.start_game()
        else:
            self.gw.reset_game()
        self._turn = 0
        self._done = False
        self._last_reward = 0.0
        self.pyboy.tick(1, True)
        self._prev = self._state()
        return self._frame()

    def step(self, buttons: list[str]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        held = [b for b in buttons if b in BUTTONS and b != "noop"][:2]  # cap at 2
        for b in held:
            self.pyboy.button_press(b)
        n_frames = self.jump_frames if "a" in held else self.move_frames
        self.pyboy.tick(n_frames, False)
        for b in held:
            self.pyboy.button_release(b)
        self.pyboy.tick(1, True)
        self._turn += 1

        cur = self._state()
        reward, terminated, event = self._reward(cur)
        truncated = self._turn >= self.max_turns
        self._prev = cur
        self._last_reward = float(reward)
        self._done = bool(terminated or truncated)
        info = dict(cur, event=event, buttons=held)
        return self._frame(), reward, self._done, info

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None


def parse_action(text: str) -> list[str]:
    m = ANSWER_RE.search(text)
    if not m:
        return ["noop"]
    try:
        parsed = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return ["noop"]
    if not isinstance(parsed, (list, tuple)):
        return ["noop"]
    buttons = [str(b).strip().lower() for b in parsed]
    valid = [b for b in buttons if b in BUTTONS]
    return valid[:2] if valid else ["noop"]


def frame_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame).convert("RGB")


def make_mario_rollout_func(
    rom_path: str,
    system_prompt: str,
    *,
    max_turns: int = 200,
    env_kwargs: dict | None = None,
):
    env_kwargs = env_kwargs or {}

    def rollout_func(prompts: list, trainer) -> dict:
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_env_mask: list[list[int]] = []
        all_images: list = []                   # per-turn PIL image list (aligned to prompt_ids)
        all_mm: list[dict] = []                 # per-turn multimodal fields (pixel_values, image_grid_thw)
        traj_returns: list[float] = []          # per-turn: the owning traj's total return
        per_turn_traj_id: list[int] = []        # group bookkeeping
        episode_returns: list[float] = []       # one R_out per episode (for standardization)
        turns_per_episode: list[int] = []       # num turns in each episode (for broadcasting)
        all_times = []
        all_len = []

        for traj_id in range(trainer.num_generations):
            env = SuperMarioLandEnv(rom_path, max_turns=max_turns, **env_kwargs)
            frame = env.reset()
            turn_examples = []
            rewards = []

            for _ in range(max_turns):
                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image", "image": frame_to_image(frame)},
                        {"type": "text", "text": "Current screen above. Decide your action."},
                    ]},
                ]
                prompt_ids, images, mm = trainer._tokenize_prompts([msg])
                comp_ids, logps, t = _generate_one(trainer, prompt_ids, images, mm)
                comp_ids = comp_ids[0]
                logps = logps[0]
                all_times.append(t)
                all_len.append(len(comp_ids))
                turn_image = images[0] if images else None
                turn_mm = {k: v for k, v in mm.items()} if mm else {}
                text = trainer.processing_class.decode(comp_ids, skip_special_tokens=True)
                action = parse_action(text)

                frame, reward, done, _info = env.step(action)
                turn_examples.append((prompt_ids[0], comp_ids, logps, turn_image, turn_mm))
                rewards.append(float(reward))
                if done:
                    break

            env.close()
            R_out = float(sum(rewards))
            episode_returns.append(R_out)
            turns_per_episode.append(len(turn_examples))
            for (p_ids, c_ids, lps, img, mm) in turn_examples:
                all_prompt_ids.append(list(p_ids))
                all_completion_ids.append(list(c_ids))
                all_logprobs.append(list(lps))
                all_env_mask.append([1] * len(c_ids))
                all_images.append([img] if img is not None else None)
                all_mm.append(mm)
                traj_returns.append(R_out)
                per_turn_traj_id.append(traj_id)

        ep = np.asarray(episode_returns, dtype=np.float64)
        ep_adv = (ep - ep.mean()) / (ep.std() + 1e-4)
        ep_adv = np.maximum(0.0, ep_adv)  # positive-advantage filtering
        all_advantages: list[float] = []
        for ep_idx, n_turns in enumerate(turns_per_episode):
            all_advantages.extend([float(ep_adv[ep_idx])] * n_turns)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_env_mask,
            "images": all_images,               # per-turn, aligned to prompt_ids
            "multimodal_fields": all_mm,        # per-turn pixel_values / image_grid_thw
            "advantages": all_advantages,       # per-turn, broadcast from episode-level
            "traj_return": traj_returns,
            "traj_id": per_turn_traj_id,
        }

    return rollout_func

def make_mario_rollout_func_v2(
    rom_path: str,
    system_prompt: str,
    *,
    max_turns: int = 200,
    env_kwargs: dict | None = None,
):
    env_kwargs = env_kwargs or {}

    def _build_msg(frame):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": frame_to_image(frame)},
                {"type": "text", "text": "Current screen above. Decide your action."},
            ]},
        ]

    def rollout_func(prompts: list, trainer) -> dict:
        n_eps = trainer.num_generations

        envs   = [SuperMarioLandEnv(rom_path, max_turns=max_turns, **env_kwargs)
                  for _ in range(n_eps)]
        frames = [env.reset() for env in envs]
        done   = [False] * n_eps
        rewards_per_ep   = [[] for _ in range(n_eps)]
        examples_per_ep  = [[] for _ in range(n_eps)]

        all_times, all_len = [], []

        for _turn in range(max_turns):
            live = [i for i in range(n_eps) if not done[i]]
            if not live:
                break

            msgs = [_build_msg(frames[i]) for i in live]
            prompt_ids, images, mm = trainer._tokenize_prompts(msgs)   # batched over live eps
            comp_ids_list, logps_list, t = _generate_one(trainer, prompt_ids, images, mm)
            all_times.append(t)

            for k, i in enumerate(live):
                comp_ids = comp_ids_list[k]
                logps    = logps_list[k]
                all_len.append(len(comp_ids))

                turn_image = _take_image(images, k)
                turn_mm    = _take_mm(mm, k)

                text   = trainer.processing_class.decode(comp_ids, skip_special_tokens=True)
                action = parse_action(text)

                frame, reward, d, _info = envs[i].step(action)
                examples_per_ep[i].append((prompt_ids[k], comp_ids, logps, turn_image, turn_mm))
                rewards_per_ep[i].append(float(reward))
                frames[i] = frame
                if d:
                    done[i] = True

        for env in envs:
            env.close()

        all_prompt_ids, all_completion_ids, all_logprobs = [], [], []
        all_env_mask, all_images, all_mm = [], [], []
        traj_returns, per_turn_traj_id = [], []
        episode_returns, turns_per_episode = [], []

        for i in range(n_eps):
            R_out = float(sum(rewards_per_ep[i]))   # undiscounted episode return (gamma=1)
            episode_returns.append(R_out)
            turns_per_episode.append(len(examples_per_ep[i]))
            for (p_ids, c_ids, lps, img, mm) in examples_per_ep[i]:
                all_prompt_ids.append(list(p_ids))
                all_completion_ids.append(list(c_ids))
                all_logprobs.append(list(lps))
                all_env_mask.append([1] * len(c_ids))
                all_images.append([img] if img is not None else None)
                all_mm.append(mm)
                traj_returns.append(R_out)
                per_turn_traj_id.append(i)

        ep = np.asarray(episode_returns, dtype=np.float64)
        ep_adv = (ep - ep.mean()) / (ep.std() + 1e-4)
        ep_adv = np.maximum(0.0, ep_adv)   # positive-advantage filtering
        all_advantages = []
        for ep_idx, n_turns in enumerate(turns_per_episode):
            all_advantages.extend([float(ep_adv[ep_idx])] * n_turns)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_env_mask,
            "images": all_images,
            "multimodal_fields": all_mm,
            "advantages": all_advantages,
            "traj_return": traj_returns,
            "traj_id": per_turn_traj_id,
        }

    return rollout_func


def _take_image(images, k):
    if not images:
        return None
    img_list = images[k]
    if not img_list:
        return None
    return img_list[0]


def _take_mm(mm, k):
    if not mm or "image_grid_thw" not in mm or "pixel_values" not in mm:
        return {}

    grid = mm["image_grid_thw"]  # (B, 3)
    pv = mm["pixel_values"]  # (sum_i rows_i, feat_dim)

    rows_per_image = grid.prod(dim=-1)  # (B)
    starts = torch.cumsum(rows_per_image, dim=0) - rows_per_image
    start = int(starts[k].item())
    end = start + int(rows_per_image[k].item())

    return {
        "pixel_values": pv[start:end],
        "image_grid_thw": grid[k:k + 1],
    }


def _generate_one(trainer, prompt_ids, images, mm):
    t0 = time.perf_counter()
    _, completion_ids, logprobs, _ = trainer.vllm_generation.generate(
        prompts=prompt_ids,
        images=images,
        num_generations=1,
        profiler=profiling_context(trainer, "vLLM.generate"),
    )
    t1 = time.perf_counter()
    # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
    logprobs = [[lp[0] for lp in seq] for seq in logprobs]
    return completion_ids, logprobs, t1-t0


def mario_reward(prompts, completions, traj_return=None, **kwargs) -> list[float]:
    if traj_return is None:
        return [0.0] * len(completions)
    return [float(r) for r in traj_return]

def plot_grpo_curves(trainer, output_dir="./curves"):
    log_history = trainer.state.log_history

    loss_steps = []
    losses = []

    reward_steps = []
    rewards = []

    for log in log_history:
        step = log.get("step")

        if "loss" in log and step is not None:
            loss_steps.append(step)
            losses.append(log["loss"])

        reward_key = None
        for key in [
            "reward",
            "rewards",
            "mean_reward",
            "reward_mean",
            "train_reward",
            "completion_reward",
            "rewards/mean",
        ]:
            if key in log:
                reward_key = key
                break

        if reward_key is not None and step is not None:
            reward_steps.append(step)
            rewards.append(log[reward_key])

    if losses:
        plt.figure()
        plt.plot(loss_steps, losses, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_curve.png", dpi=200)
        plt.show()
    else:
        print("No loss values found in trainer.state.log_history")

    if rewards:
        plt.figure()
        plt.plot(reward_steps, rewards, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reward_curve.png", dpi=200)
        plt.show()
    else:
        print("No reward values found in trainer.state.log_history")
        print("Available log keys:")
        all_keys = sorted({k for log in log_history for k in log.keys()})
        print(all_keys)