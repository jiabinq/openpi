# AutoDL `lerobot` Download Fix (Proxy‑Friendly, Consolidated)

This guide is the single source of truth for installing `lerobot` (and `dlimp`) on AutoDL when `git clone` or PyPI is flaky. Follow the blocks in order. If a block says “Once per shell”, run it at the start of each new shell.

## 0) Proxy bootstrap (Once per shell)

```bash
# Load AutoDL network settings
source /etc/network_turbo

# Export uppercase variants for picky tools
for v in http_proxy https_proxy no_proxy; do eval "export ${v^^}=
\$$v"; done

# Quick sanity checks
env | grep -i '_proxy' || true
curl -I https://pypi.org/simple || true

# Optional: use a fast mirror by default for uv/pip
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

Notes:
- Ensure `NO_PROXY` does NOT include `pypi.org` or `files.pythonhosted.org`.
- For private nets, you can add: `10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,localhost,127.0.0.1,::1`.

## 1) Download archives (run if not already downloaded)

```bash
# lerobot @ 0cf8648
curl -L \
  https://codeload.github.com/huggingface/lerobot/tar.gz/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  -o /root/autodl-tmp/lerobot.tgz

# dlimp @ ad72ce3
curl -L \
  https://codeload.github.com/kvablack/dlimp/tar.gz/ad72ce3a9b414db2185bc0b38461d4101a65477a \
  -o /root/autodl-tmp/dlimp.tgz
```

## 2) Install `lerobot` (fast path)

```bash
uv pip install \
  --index-url "${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --extra-index-url https://pypi.org/simple \
  /root/autodl-tmp/lerobot.tgz
```

If this succeeds, skip to Step 4. If it fails due to a PyPI wheel (e.g., `pytz`), pick ONE of the fallbacks in Step 3 and then retry the same `uv pip install` above.

## 3) If PyPI wheels fail (choose ONE)

3A) Seed only the missing wheel (e.g., `pytz==2025.2`):
```bash
curl -L \
  https://pypi.tuna.tsinghua.edu.cn/packages/py3/p/pytz/pytz-2025.2-py2.py3-none-any.whl \
  -o /root/autodl-tmp/pytz-2025.2-py2.py3-none-any.whl
uv pip install /root/autodl-tmp/pytz-2025.2-py2.py3-none-any.whl
```

3B) Build a small offline wheelhouse for `lerobot`:
```bash
mkdir -p /root/autodl-tmp/wheelhouse
uv pip download \
  --index-url "${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --dest /root/autodl-tmp/wheelhouse \
  /root/autodl-tmp/lerobot.tgz
uv pip install \
  --no-index --find-links=/root/autodl-tmp/wheelhouse \
  /root/autodl-tmp/lerobot.tgz
```

## 4) Install `dlimp` (same pattern)

```bash
uv pip install \
  --index-url "${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --extra-index-url https://pypi.org/simple \
  /root/autodl-tmp/dlimp.tgz
```

If it fails similarly, apply Step 3 with `dlimp.tgz` instead of `lerobot.tgz`.

## 5) Finish dependency resolution

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Troubleshooting tips:
- Verbose install to confirm proxy usage: `uv -v pip install pytz==2025.2`
- Git behind proxy (for clones):
  ```bash
  git config --global http.proxy "$HTTP_PROXY"
  git config --global https.proxy "$HTTPS_PROXY"
  ```
