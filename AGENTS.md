# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/openpi` (core libraries and models).
- Scripts: `scripts/` (CLI entry points like `train.py`, `serve_policy.py`, `compute_norm_stats.py`).
- Packages: `packages/*` (workspace members vendored via `pyproject.toml`).
- Examples: `examples/` (platform-specific how-tos and datasets).
- Docs & Assets: `docs/`, `assets/`; Checkpoints: `checkpoints/`.
- Third‑party and data vendors: `third_party/`.
- Tests live next to code in `src/`, `scripts/`, and `packages/` (see `pytest` `testpaths`).

## Build, Test, and Development Commands
- Environment: `GIT_LFS_SKIP_SMUDGE=1 uv sync && uv pip install -e .` (install deps and this repo editable).
- Lint (fix): `uv run ruff check . --fix` and format: `uv run ruff format`.
- Tests: `uv run pytest -q` (respects markers; GPU-heavy tests may be marked `@pytest.mark.manual`).
- Pre-commit: `uv run pre-commit install` then `uv run pre-commit run -a`.
- Run training: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name>`.
- Serve policy: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt_dir>`.

## Coding Style & Naming Conventions
- Python 3.11+, 4‑space indent, type hints required for public APIs.
- Max line length 120; imports single-line sorted (Ruff Isort settings).
- Modules/functions: `snake_case`; classes: `CapWords`; constants: `UPPER_SNAKE_CASE`.
- Prefer logging in library code; `print` acceptable in scripts.
- Keep JAX/Flax code pure and functional; avoid side effects in hot paths.

## Testing Guidelines
- Framework: `pytest`. Place `test_*.py` beside the code under test.
- Use fixtures over ad‑hoc setup; parametrize where possible.
- Mark long/GPU tests as `@pytest.mark.manual` so CI or quick runs can skip.
- Aim to cover config edge cases and data transforms (e.g., normalization stats, tokenizers).

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (≤72 chars), descriptive body when needed. Reference issues/PRs (e.g., `Fix temperature sampling (#550)`).
- PRs: include summary, motivation, before/after notes, and run commands to reproduce. Link issues, attach logs or screenshots for examples.
- Requirements: green `pre-commit`, `ruff`, and `pytest`; update docs (`README.md`, `examples/*`, or `docs/`) when changing behavior.

## Security & Configuration Tips
- GPUs: set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for JAX memory usage.
- W&B: export `WANDB_API_KEY` if logging to Weights & Biases.
- Large deps via submodules/LFS: always use `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
