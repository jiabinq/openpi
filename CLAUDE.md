# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenPi is an open-source robotics framework by Physical Intelligence for Vision-Language-Action (VLA) models. It provides both diffusion-based (π₀) and autoregressive (π₀-FAST) models for robot control.

## Key Commands

### Setup and Dependencies
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or update submodules after cloning
git submodule update --init --recursive

# Install dependencies with uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest path/to/test_file.py

# Run tests excluding manual tests
uv run pytest -m "not manual"
```

### Code Quality
```bash
# Run linter and formatter
uv run ruff check
uv run ruff format

# Fix linting issues automatically
uv run ruff check --fix
```

### Training Commands
```bash
# Compute normalization statistics before training
uv run scripts/compute_norm_stats.py --config-name CONFIG_NAME

# Run training (set memory fraction for optimal GPU usage)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py CONFIG_NAME --exp-name=EXPERIMENT_NAME --overwrite

# Serve a trained policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=CONFIG_NAME --policy.dir=CHECKPOINT_PATH
```

## Architecture Overview

### Core Components (`src/openpi/`)

- **models/**: VLA model implementations
  - `pi0.py`, `pi0_fast.py`: Core diffusion and autoregressive models
  - `gemma.py`, `gemma_fast.py`: Language model components
  - `siglip.py`, `vit.py`: Vision encoders
  - `tokenizer.py`: Action tokenization for FAST models
  - `lora.py`: LoRA adaptation for fine-tuning

- **policies/**: Robot-specific policy implementations
  - Each policy (e.g., `aloha_policy.py`, `droid_policy.py`) defines:
    - Input/output data mappings
    - Normalization configurations
    - Robot-specific action spaces

- **training/**: Training infrastructure
  - `config.py`: Defines `TrainConfig` and dataset-specific configs
  - `data_loader.py`: Handles LeRobot dataset loading
  - `weight_loaders.py`: Manages pre-trained checkpoint loading

- **serving/**: Model serving
  - `websocket_policy_server.py`: WebSocket-based remote inference

### Key Design Patterns

1. **Config-Driven**: All training and inference configurations are defined in `training/config.py` with a registry system (`CONFIG_TABLE`)

2. **Policy Abstraction**: Each robot platform has its own policy class inheriting from `Policy` base class, handling:
   - Data preprocessing/postprocessing
   - Action/observation normalization
   - Robot-specific constraints

3. **Remote Inference**: Models can run on separate servers and stream actions via WebSocket to robots

4. **Checkpoint Management**: Automatic downloading from Google Cloud Storage with local caching in `~/.cache/openpi`

### Data Flow

1. **Training**: Raw data → LeRobot dataset → DataLoader → Model → Checkpoints
2. **Inference**: Robot observations → Policy preprocessing → Model → Policy postprocessing → Robot actions

### Key Configurations

- **Model Variants**: π₀ (diffusion, 50 denoising steps) and π₀-FAST (autoregressive, single forward pass)
- **Training Modes**: Full fine-tuning or LoRA (requires less GPU memory)
- **Normalization**: Pre-computed statistics for state/action normalization, can be reloaded from pre-training

### Testing Approach

- Tests are located alongside source files (`*_test.py`)
- Manual tests marked with `@pytest.mark.manual` for hardware-dependent tests
- Run tests with `uv run pytest`