# OpenPi Installation Summary

## Date: 2025-08-20

### Issue Encountered
When trying to install OpenPi dependencies using the standard commands:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The installation failed with the following error:
- `mujoco==2.3.7` failed to build because it required the `MUJOCO_PATH` environment variable
- This older version of mujoco was being installed as a dependency of `gym-aloha==0.1.1`

### Solution Steps

1. **Initial attempt with system Python (3.13.2)**
   - Failed due to mujoco build error
   - Tried installing mujoco separately with `--no-build-isolation` flag
   - Successfully installed newer mujoco (3.3.5) but the main installation still failed

2. **Created Python 3.11 virtual environment**
   ```bash
   uv venv --python 3.11
   ```
   - This created a `.venv` directory with Python 3.11.13

3. **Successful installation in venv**
   ```bash
   source .venv/bin/activate
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   ```
   - Installation completed successfully in the Python 3.11 environment

### Key Packages Installed
- openpi==0.1.0 (editable install)
- jax==0.5.3 (with CUDA support)
- torch==2.7.0
- flax==0.10.2
- mujoco (resolved dependency issues)
- And many other dependencies

### Lessons Learned
- Python version compatibility matters - Python 3.11 worked while 3.13 had issues
- The mujoco dependency conflict was resolved by using a proper virtual environment
- Using `uv` for both environment creation and package management simplified the process

### To Activate Environment
```bash
source .venv/bin/activate
```