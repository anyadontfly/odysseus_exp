import argparse
import os
import time

from datasets import Dataset
from transformers import AutoProcessor

from trl import GRPOConfig, GRPOTrainer
from utils import make_mario_rollout_func, mario_reward, SYSTEM_PROMPT

import matplotlib.pyplot as plt


ROM_PATH = "sml.gb"
OUTPUT_DIR = "./odysseus_ckpts"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--steps_per_generation", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=28)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--use_liger_kernel", action='store_true')
    return parser.parse_args()


def build_dataset(n_rows: int) -> Dataset:
    prompt = [{"role": "", "content": ""}]
    return Dataset.from_list([{"prompt": prompt} for _ in range(n_rows)])


def main() -> None:
    args = parse_args()

    # actual batch size ≈ per_device_train_batch_size * max_turns / steps_per_generation
    generation_batch_size = args.per_device_train_batch_size * args.steps_per_generation

    proc_kwargs = {"trust_remote_code": True}
    processing_class = AutoProcessor.from_pretrained(args.model, **proc_kwargs)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # resume_from_checkpoint=OUTPUT_DIR,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        steps_per_generation=args.steps_per_generation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        # report_to="wandb",
        # log_completions=True,
        use_vllm=True,
        vllm_mode="server",
        bf16=True,
        use_liger_kernel=args.use_liger_kernel,
        gradient_checkpointing=args.gradient_checkpointing,
        num_completions_to_print=0,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        # torch_empty_cache_steps=1,
    )

    rollout_func = make_mario_rollout_func(
        ROM_PATH,
        SYSTEM_PROMPT,
        max_turns=args.max_turns,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=mario_reward,
        args=config,
        train_dataset=build_dataset(generation_batch_size),
        processing_class=processing_class,
        rollout_func=rollout_func,
    )

    trainer.train()


if __name__ == "__main__":
    main()
