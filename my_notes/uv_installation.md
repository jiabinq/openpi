# UV Installation Guide

UV is a fast Python package and project manager that serves as a modern replacement for pip, pip-tools, pipx, poetry, pyenv, and virtualenv. This guide covers how to install UV on your system.

## System Requirements

- **Operating System**: Linux (Ubuntu 22.04 tested), macOS, or Windows
- **Python**: Not required for installation (UV manages Python versions)
- **Architecture**: x86_64 or ARM64

## Installation Methods

### 1. Standalone Installer (Recommended for Linux/macOS)

The easiest way to install UV on Linux or macOS is using the standalone installer:

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

This will:
- Download the latest UV binary
- Install it to `~/.cargo/bin`
- Add the installation directory to your PATH (you may need to restart your shell)

### 2. Windows Installation

For Windows, use PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Alternative Installation Methods

#### Using pipx (Recommended if you already have pipx)
```bash
pipx install uv
```

#### Using pip (Not recommended for general use)
```bash
pip install uv
```

#### Package Managers

**Homebrew (macOS/Linux):**
```bash
brew install uv
```

**WinGet (Windows):**
```powershell
winget install --id=astral-sh.uv -e
```

**Scoop (Windows):**
```powershell
scoop install main/uv
```

#### Docker
UV is available as a Docker image:
```bash
docker pull ghcr.io/astral-sh/uv
```

#### From Source (Requires Rust)
```bash
# Install Rust first if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install UV from source
cargo install --git https://github.com/astral-sh/uv uv
```

## Verification

After installation, verify UV is working:

```bash
# Check UV version
uv --version

# Check installation location
which uv
```

## Post-Installation Setup

1. **Restart your shell** or source your profile:
   ```bash
   source ~/.bashrc  # or ~/.zshrc for zsh
   ```

2. **Verify PATH**: Ensure UV is in your PATH:
   ```bash
   echo $PATH | grep -E "(cargo/bin|\.local/bin)"
   ```

## Usage with OpenPi

Once UV is installed, you can use it to set up the OpenPi environment:

```bash
# Clone the repository with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# Set up the environment with UV
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Upgrading UV

To upgrade UV to the latest version:

```bash
# If installed with standalone installer
uv self update

# If installed with pipx
pipx upgrade uv

# If installed with Homebrew
brew upgrade uv

# Other package managers
# Use the respective upgrade command
```

## Troubleshooting

### Common Issues

1. **"Command not found" after installation**
   - Solution: Restart your shell or manually add UV to PATH:
     ```bash
     export PATH="$HOME/.cargo/bin:$PATH"
     ```

2. **Permission denied during installation**
   - Solution: Ensure you have write permissions to the installation directory
   - Do NOT use sudo with the curl/wget installer

3. **SSL/TLS errors during download**
   - Solution: Update your system's certificates:
     ```bash
     sudo apt-get update && sudo apt-get install ca-certificates
     ```

4. **UV commands fail with Python errors**
   - Solution: UV manages its own Python installations. Try:
     ```bash
     uv python install 3.11
     ```

### OpenPi-Specific Issues

1. **"uv sync" fails with dependency conflicts**
   - Solution: Remove the virtual environment and retry:
     ```bash
     rm -rf .venv
     uv sync
     ```

2. **Git LFS errors**
   - Solution: Always use `GIT_LFS_SKIP_SMUDGE=1` when running UV commands for OpenPi

## Additional Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
- [OpenPi Repository](https://github.com/Physical-Intelligence/openpi)

## Notes for OpenPi Development

- UV is the recommended package manager for OpenPi
- Python 3.11+ is required for OpenPi
- UV will automatically manage virtual environments
- The project uses `pyproject.toml` for dependency management
- Lock file (`uv.lock`) ensures reproducible installations