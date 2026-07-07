## 0. Download the modifed trl source

```bash
git clone https://github.com/anyadontfly/trl.git
mv trl trl_src
cd trl_src
git checkout puyuan-dev-1.5.1
cd ..
```

## 1. Build the image

```bash
bash scripts/build_image.sh
```

Produces `/projects/bcjw/pyao3/trl-dev.sif`.

## 2. Run a job

Batch job:

```bash
sbatch submit_train.sh [script to run]
```

Interactive:

```bash
bash scripts/run_container.sh
[script to run]
```

Pick an experiment by running the matching launcher in `scripts/`, e.g.
`run-4b-max_turns10-group_sz4-bsz2-…` (Qwen3-4B, max 10 turns, group size 4,
batch size 2).

## Notes

- GB file: place your game gb file to current dir and rename it to `sml.gb`
- Outputs: checkpoints → `odysseus_ckpts/`, logs → `logs/` + `slurm_logs/`,
  curves → `curves/`, W&B → `wandb/`.