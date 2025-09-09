Autodl: Persist caches and data on data drive
============================================

Goal
----
- Keep datasets, checkpoints, and caches off the 30G system disk and on the data drive.
- Make cache locations persistent across sessions via `~/.bashrc`.

Base paths
----------
- Data base: `/root/autodl-tmp/openpi-data`
- Recommended layout:
  - Datasets: `/root/autodl-tmp/openpi-data/datasets`
  - Checkpoints: `/root/autodl-tmp/openpi-data/checkpoints`
  - Caches: `/root/autodl-tmp/openpi-data/{hf,cache,wandb,uv-cache,torch,pip-cache,cuda-cache,matplotlib}`

1) Create directories (one‑time)
--------------------------------
```bash
mkdir -p /root/autodl-tmp/openpi-data/{datasets,checkpoints,cache,hf,wandb,uv-cache,torch,pip-cache,cuda-cache,matplotlib}
```

2) Persist env vars in ~/.bashrc (one‑time)
-------------------------------------------
Use a here‑doc to append the block below. Important: type/paste the closing `EOF` on its own line to finish.

```bash
cat >> ~/.bashrc <<'EOF'
# OpenPI data-drive caches (Autodl)
export OPENPI_DATA_BASE=/root/autodl-tmp/openpi-data
export HF_HOME="$OPENPI_DATA_BASE/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$OPENPI_DATA_BASE/cache"
export WANDB_DIR="$OPENPI_DATA_BASE/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export UV_CACHE_DIR="$OPENPI_DATA_BASE/uv-cache"
export TORCH_HOME="$OPENPI_DATA_BASE/torch"
export PIP_CACHE_DIR="$OPENPI_DATA_BASE/pip-cache"
export CUDA_CACHE_PATH="$OPENPI_DATA_BASE/cuda-cache"
export MPLCONFIGDIR="$OPENPI_DATA_BASE/matplotlib"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
[ -f /etc/network_turbo ] && source /etc/network_turbo
EOF
```

3) Reload current shell and verify
----------------------------------
```bash
source ~/.bashrc
env | egrep 'OPENPI_DATA_BASE|HF_HOME|HUGGINGFACE_HUB_CACHE|HF_DATASETS_CACHE|XDG_CACHE_HOME|UV_CACHE_DIR|TORCH_HOME|CUDA_CACHE_PATH|HF_HUB_ENABLE_HF_TRANSFER|UV_LINK_MODE'
df -h / /root/autodl-tmp
```

Optional: project‑local symlinks
--------------------------------
If the repo expects `checkpoints/` or `datasets/` inside the project, point them at the data drive:

```bash
ln -sfn /root/autodl-tmp/openpi-data/checkpoints /root/autodl-tmp/openpi/checkpoints
ln -sfn /root/autodl-tmp/openpi-data/datasets    /root/autodl-tmp/openpi/datasets
```

Notes
-----
- Network accelerator: `source /etc/network_turbo` (already added to `~/.bashrc`).
- UV hardlink warning: using `UV_LINK_MODE=copy` avoids cross‑filesystem hardlink issues.
- HF downloads: `HF_HUB_ENABLE_HF_TRANSFER=1` speeds up transfers.
- Why this is persistent: each new shell sources `~/.bashrc`, so tools (HF, UV, Torch, CUDA) respect these paths
  and keep caches on the data drive.

