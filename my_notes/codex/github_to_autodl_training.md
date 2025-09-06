# Push to GitHub and Train on AutoDL

## Goal
Document the exact steps to push local changes (shape‑aware weight merge, config fixes) to GitHub and run training on AutoDL.

## 1) Prepare and Push Locally
- Create branch: `git checkout -b feat/shape-aware-weights`
- Lint/format: `uv run ruff check . --fix && uv run ruff format`
- Verify syntax: `python -m py_compile src/openpi/training/{config,weight_loaders}.py`
- Stage + commit:
  - `git add src/openpi/training/weight_loaders.py src/openpi/training/config.py AGENTS.md CHANGELOG.md my_notes/codex/weight_loader_shape_aware_merge.md`
  - `git commit -m "weights: shape-aware checkpoint merge for action_dim!=32; docs: changelog + notes; config: import fix"`
- Push: `git push -u origin feat/shape-aware-weights`
- Open PR and merge to `main` (or push directly if desired).

## 2) Clone on AutoDL
- System deps:
  - `sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 libglib2.0-0 git-lfs`
- UV install: `curl -Ls https://astral.sh/uv/install.sh | sh && exec $SHELL`
- Clone with submodules:
  - `git clone --recurse-submodules <repo-url> -b main && cd openpi`
- Use fast disk for cache: `export OPENPI_DATA_HOME=/data/.cache/openpi`
- Python deps:
  - `GIT_LFS_SKIP_SMUDGE=1 uv sync`
  - `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
- (Private HF) Login: `uv run huggingface-cli login`

## 3) Compute Stats and Train
- Norm stats: `uv run scripts/compute_norm_stats.py --config-name pi0_clear_tray_fine_tune`
- Train (resume or fresh):
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_clear_tray_fine_tune --exp-name=my_first_pi0jax_run --resume`
  - Use `--overwrite` for a fresh run.

## 4) Notes & Troubleshooting
- Expected warnings: shape‑aware loader may log skipped params for action heads (32→6). This is intended.
- OOM:
  - Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
  - On multi‑GPU, try `--fsdp-devices=<num_gpus>`.
  - Reduce batch size if increased from default.
- Verify: W&B shows images from first batch and steady loss logging.
- Optional tag: `git tag -a v0.1.1-shape-merge -m "Shape-aware weight merge" && git push --tags`
