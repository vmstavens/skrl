# Repository Cleanup

Date: 2026-06-15

## What changed

- Removed generated Python cache directories (`__pycache__`) from the package, test, utility, asset, dataset, simulation, plotting, and experiment trees.
- Removed local experiment/runtime output directories:
  - `wandb/`
  - `runs/`
  - `.runs/`, `results/`, and `media/` directories under `testing/experiments/`
- Removed generated local media and plot output:
  - root rollout videos and output/plot PNGs
  - `imgs/`
  - `media/`
  - `plotting/plots/`
  - `testing/media/`
  - rollout GIFs and mocap metric PNGs under `testing/envs/`
- Removed the local MuJoCo runtime log `MUJOCO_LOG.TXT`.
- Updated `.gitignore` so generated artifacts stay out of Git:
  - local Codex/agent metadata (`.codex`, `.codex/`, `.agents`, `.agents/`)
  - experiment run directories (`runs/`, `testing/experiments/.runs/`, nested `.runs/`)
  - rendered media and rollout artifacts (`*.mp4`, `*.gif`, rollout GIFs, mocap metric images)
  - local media/image output directories (`media/`, `imgs/`, `plotting/plots/`, `testing/media/`)
  - runtime logs (`MUJOCO_LOG.TXT`, `rollout_log*.txt`)
  - generated root-level images (`/*.png`, including `output*.png`)

## What was intentionally left alone

- Existing modified tracked source files were not reverted or edited.
- Untracked source-like files and experiment code were left in place for review.
- Project assets under `assets/` were not ignored wholesale, so real assets can still be added intentionally.

## Follow-up

Run `git status --short` before committing. The expected cleanup-related files to stage are:

- `.gitignore`
- `CLEANUP.md`
