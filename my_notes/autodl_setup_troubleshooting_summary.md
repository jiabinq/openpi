# AutoDL Network Hiccup: `uv sync` fails to clone `lerobot` (Beginner‑Friendly)

## Symptom
- Running `GIT_LFS_SKIP_SMUDGE=1 uv sync` fails with an error like:
  - `failed to clone into ... failed to fetch commit ... Recv failure: Connection reset by peer`

## What This Means
- `uv sync` tries to download a dependency (`lerobot`) from GitHub at a specific commit.
- The container can reach GitHub (e.g., `curl -I https://github.com/huggingface/lerobot` works), but `git fetch` is getting reset mid‑transfer. This is a transient network or proxy/CDN issue.

## Quick Checks
- Confirm internet egress: `curl -I https://github.com/huggingface/lerobot`
- Optional: try `git ls-remote https://github.com/huggingface/lerobot`

## Solutions (try in order)
1) Retry `uv sync` (sometimes it’s transient):
   - `GIT_LFS_SKIP_SMUDGE=1 uv sync`
2) Force Git HTTP/1.1 (more robust through proxies), then retry:
   - `git config --global http.version HTTP/1.1`
   - `GIT_LFS_SKIP_SMUDGE=1 uv sync`
3) Add resilience (if still failing), then retry:
   - `git config --global http.postBuffer 524288000`
   - `git config --global http.lowSpeedLimit 0`
   - `git config --global http.lowSpeedTime 0`
   - `git config --global http.retry 5`
   - `GIT_LFS_SKIP_SMUDGE=1 uv sync`
4) Tarball fallback (bypasses `git fetch`):
   - `curl -L https://codeload.github.com/huggingface/lerobot/tar.gz/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 -o /root/autodl-tmp/lerobot.tgz`
   - `uv pip install /root/autodl-tmp/lerobot.tgz`
   - `curl -L https://codeload.github.com/kvablack/dlimp/tar.gz/ad72ce3a9b414db2185bc0b38461d4101a65477a -o /root/autodl-tmp/dlimp.tgz`
   - `uv pip install /root/autodl-tmp/dlimp.tgz`
   - Then: `GIT_LFS_SKIP_SMUDGE=1 uv sync`
5) If the node lacks a GPU and you plan to train, consider moving to a GPU‑backed instance (CUDA 12.x). These nodes often have better egress, and you’ll need GPU later anyway.

## After It Works
- Finish setup:
  - `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
  - Use fast disk for caches: `export OPENPI_DATA_HOME=/root/autodl-tmp/.cache/openpi`
- When a GPU is available, verify JAX: `uv run python -c "import jax; print(jax.__version__, jax.devices())"`
- Continue with training steps as usual.
