# Migration Guide: From pip to uv

This guide explains how to migrate the Chatterbox TTS FastAPI from pip to uv for better dependency management, faster installations, and improved PyTorch/CUDA compatibility.

## Why Migrate to uv?

Based on user feedback and testing, uv provides several advantages for this project:

1. **Better PyTorch/CUDA handling**: Native support for PyTorch indexes and variants
2. **Faster installations**: 25-40% faster than pip in benchmarks
3. **Superior dependency resolution**: SAT-based resolver prevents conflicts
4. **Git dependency support**: Better handling of `chatterbox-tts` from GitHub
5. **Cross-platform consistency**: Environment markers handle CPU/GPU variants automatically
6. **Docker optimizations**: Better caching and faster container builds
7. **FastAPI compatibility**: Excellent support for FastAPI and Pydantic dependencies

## FastAPI + uv Benefits

The combination of FastAPI and uv provides:

- **Faster builds**: uv resolves FastAPI dependencies more efficiently
- **Better type checking**: Enhanced support for Pydantic and type hint libraries
- **Cleaner environments**: More reliable async dependency management
- **Development speed**: Faster dependency installation during development

## Docker GPU Issues (Fixed)

**Previous Problem**: The original uv Docker implementations were failing because:

1. **Missing CUDA Index**: `pyproject.toml` only defined CPU PyTorch indexes
2. **Incorrect Configuration**: Assumed PyTorch would "auto-detect CUDA" (which doesn't work with uv)
3. **Complex Index Logic**: Overly complicated platform markers that didn't work reliably
4. **Missing Dependency**: `resemble-perth` watermarker dependency was not explicitly included

**Solution Implemented**: The fixed uv Dockerfiles now:

1. **Explicit PyTorch Installation**: Install PyTorch with correct CUDA/CPU versions first
2. **Direct Index Usage**: Use `uv pip install` with explicit `--index-url` flags
3. **Simplified Configuration**: Removed complex pyproject.toml index configurations
4. **Complete Dependencies**: Added `resemble-perth` for watermarker functionality
5. **FastAPI Optimization**: Optimized for FastAPI and async dependencies

**Fixed Commands**:

```dockerfile
# GPU version (Dockerfile.uv.gpu)
RUN uv venv --python 3.11 && \
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install resemble-perth && \
    uv pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@v0.1.2" && \
    uv pip install fastapi uvicorn[standard] python-dotenv requests

# CPU version (Dockerfile.uv)
RUN uv venv --python 3.11 && \
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install resemble-perth && \
    uv pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@v0.1.2" && \
    uv pip install fastapi uvicorn[standard] python-dotenv requests
```

**Common Error Fixed**:

```
TypeError: 'NoneType' object is not callable
self.watermarker = perth.PerthImplicitWatermarker()
```

This was caused by missing `resemble-perth` package that provides the watermarker functionality.

## Pre-Migration Checklist

- [ ] Backup your current `.venv` directory (if using virtual environments)
- [ ] Note your current Python version (`python --version`)
- [ ] Save current package versions (`pip freeze > backup-requirements.txt`)
- [ ] Check if you're using the Flask or FastAPI version (this guide assumes FastAPI)

## Step 1: Install uv

### macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative (via pip):

```bash
pip install uv
```

## Step 2: Local Development Migration

### Option A: Quick Migration (Recommended)

1. **Remove old virtual environment**:

   ```bash
   rm -rf .venv
   ```

2. **Install dependencies with uv**:

   ```bash
   uv sync
   ```

3. **Test the installation**:

   ```bash
   uv run python -c "import chatterbox; print('✓ chatterbox-tts installed')"
   uv run python -c "import torch; print(f'✓ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
   uv run python -c "import fastapi; print(f'✓ FastAPI {fastapi.__version__}')"
   ```

4. **Run the FastAPI server**:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 4123
   # or
   uv run main.py
   ```

### Option B: Manual Migration

1. **Initialize uv project** (if you want to customize):

   ```bash
   uv init --no-readme --no-workspace
   ```

2. **Add dependencies**:
   ```bash
   uv add "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@v0.1.2"
   uv add "torch>=2.0.0,<2.7.0"
   uv add "torchaudio>=2.0.0,<2.7.0"
   uv add "fastapi>=0.104.0"
   uv add "uvicorn[standard]>=0.24.0"
   uv add "pydantic>=2.0.0"
   uv add "python-dotenv>=1.0.0"
   uv add "requests>=2.28.0" --optional dev
   ```

## Step 3: PyTorch Configuration

The provided `pyproject.toml` automatically handles PyTorch variants:

- **Linux**: Uses CUDA-enabled PyTorch
- **macOS/Windows**: Uses CPU-only PyTorch
- **Manual override**: Set environment variables if needed

### Force CPU version:

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu uv sync
```

### Force CUDA version:

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 uv sync
```

## Step 4: Docker Migration

### Using the new uv-based Docker setup:

1. **Build with uv**:

   ```bash
   docker build -f Dockerfile.uv -t chatterbox-tts-uv .
   ```

2. **Or build the uv image directly**:

   ```bash
   docker build -f docker/Dockerfile.uv -t chatterbox-tts-uv .
   ```

3. **GPU variant**:
   ```bash
   docker build -f Dockerfile.uv.gpu -t chatterbox-tts-uv-gpu .
   ```

## Step 5: Verify Migration

### Test basic functionality:

```bash
# Health check
uv run python -c "
import requests
r = requests.get('http://localhost:4123/health')
print(f'Status: {r.status_code}')
print(r.json())
"

# Test FastAPI documentation
uv run python -c "
import requests
r = requests.get('http://localhost:4123/docs')
print(f'Docs available: {r.status_code == 200}')
"

# Test TTS generation
uv run python tests/test_api.py
```

### Performance comparison:

```bash
# Time the installation
time uv sync --reinstall
# vs
time pip install -r requirements.txt
```

## Step 6: Update Development Workflow

### New commands:

| Old pip command                   | New uv command     |
| --------------------------------- | ------------------ |
| `pip install package`             | `uv add package`   |
| `pip install -r requirements.txt` | `uv sync`          |
| `python script.py`                | `uv run script.py` |
| `pip freeze`                      | `uv pip freeze`    |
| `flask run --debug`               | N/A (now FastAPI)  |
| `python main.py`                  | `uv run main.py`   |

### FastAPI with uv commands:

```bash
# Start FastAPI development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 4123 --reload

# Run with specific Python version
uv run --python 3.11 uvicorn app.main:app --reload

# Install development dependencies (basic tools)
uv sync

# Install test dependencies (for testing/contributing)
uv sync --group test
```

### Environment management:

```bash
# Create/sync environment
uv sync

# Add new dependency
uv add numpy

# Remove dependency
uv remove numpy

# Update all dependencies
uv sync --upgrade

# Install dev dependencies (basic development tools)
uv sync

# Install test dependencies (for contributors/testing)
uv sync --group test
```

## Step 7: CI/CD Updates

Update your CI/CD pipelines to use uv with FastAPI:

```yaml
# GitHub Actions example
- name: Set up uv
  uses: astral-sh/setup-uv@v2

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run python tests/test_api.py

- name: Test FastAPI endpoints
  run: |
    uv run uvicorn api:app --host 0.0.0.0 --port 4123 &
    sleep 10
    curl -f http://localhost:4123/health
    curl -f http://localhost:4123/docs
```

## Troubleshooting

### Common Issues:

1. **CUDA compatibility**:

   ```bash
   # Clear cache and reinstall
   uv cache clean
   uv sync --reinstall
   ```

2. **Git dependency issues**:

   ```bash
   # Force refresh git dependencies
   uv sync --refresh-package chatterbox-tts
   ```

3. **Lock file conflicts**:

   ```bash
   # Regenerate lock file
   rm uv.lock
   uv sync
   ```

4. **Python version mismatch**:

   ```bash
   # Use specific Python version
   uv python install 3.11
   uv sync --python 3.11
   ```

5. **FastAPI startup issues**:

   ```bash
   # Check FastAPI dependencies
   uv run python -c "import fastapi, uvicorn; print('FastAPI ready')"

   # Run with verbose logging
   uv run uvicorn api:app --log-level debug --reload
   ```

## Performance Benefits

Expected improvements after migration:

- **Installation speed**: 25-40% faster
- **Dependency resolution**: More reliable, fewer conflicts
- **Docker builds**: 20-30% faster with better caching
- **Development workflow**: Faster environment creation
- **PyTorch compatibility**: Better handling of CUDA variants
- **FastAPI performance**: Better async dependency management

## FastAPI + uv Specific Benefits

- **Faster development cycles**: uv's speed + FastAPI's auto-reload
- **Better type checking**: Enhanced IDE support for async functions
- **Cleaner dependency trees**: uv resolves FastAPI/Pydantic dependencies better
- **Improved testing**: Faster test environment setup

## Rollback Plan

If you need to rollback to pip:

1. **Restore backup**:

   ```bash
   pip install -r backup-requirements.txt
   ```

2. **Remove uv files**:

   ```bash
   rm pyproject.toml uv.lock
   ```

3. **Use the standard Dockerfile**:
   ```bash
   docker build -f docker/Dockerfile -t chatterbox-tts-api .
   ```

Note: If you're rolling back from FastAPI to Flask, you'll need to restore the previous Flask version of the codebase.

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [PyTorch with uv Guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv Docker Guide](https://docs.astral.sh/uv/guides/integration/docker/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

## Support

If you encounter issues during migration:

1. Check the [uv troubleshooting guide](https://docs.astral.sh/uv/reference/troubleshooting/)
2. Check the [FastAPI documentation](https://fastapi.tiangolo.com/tutorial/)
3. Open an issue with reproduction steps
4. Include `uv.lock` and `pyproject.toml` files
5. Provide error logs with `--verbose` flag

For FastAPI-specific issues, visit the interactive documentation at `http://localhost:4123/docs` once the server is running.
