# Task 07: Update Project Dependencies

## High-Level Objective
Add required dependencies for the STT processor and audio processing utilities to the project configuration.

## Core Direction
Update pyproject.toml to include HuggingFace transformers, PyTorch, and audio processing libraries while maintaining version compatibility and proper dependency management.

## Dependencies
- Can be done in parallel with implementation tasks
- Should be completed before testing Task 06 (STT Processor)

## Prerequisites
- Understanding of Python packaging and dependency management
- Knowledge of required libraries for STT processing

## Acceptance Criteria

### 1. Core Dependencies
Add to `[project.dependencies]`:
- [ ] `transformers>=4.40.0` - HuggingFace transformers for model loading
- [ ] `torch>=2.2.0` - PyTorch for model inference
- [ ] `torchaudio>=2.2.0` - PyTorch audio utilities
- [ ] `soundfile>=0.12.0` - Audio file I/O
- [ ] `librosa>=0.10.0` - Audio processing utilities (optional but recommended)

### 2. Optional Dependencies
Consider adding to `[project.optional-dependencies]`:
- [ ] Create `audio` extra for audio-specific dependencies
- [ ] Create `stt` extra for speech-to-text specific dependencies
- [ ] Example: `pip install ai-assistant[stt]` for STT support

### 3. Development Dependencies
Update `[project.optional-dependencies.dev]` if needed:
- [ ] `pytest-timeout>=2.2.0` - For handling slow model loading in tests
- [ ] `pytest-mock>=3.12.0` - For mocking model inference

### 4. Platform Considerations
- [ ] Add notes about CUDA availability for GPU support
- [ ] Consider MPS (Metal Performance Shaders) for Apple Silicon
- [ ] Document CPU-only installation if needed

### 5. Version Constraints
- [ ] Ensure PyTorch version compatible with transformers
- [ ] Ensure numpy version compatible with all dependencies
- [ ] Test dependency resolution

## Implementation Notes

### Updated pyproject.toml Structure
```toml
[project]
name = "ai-assistant"
version = "0.1.0"
description = "Multi-threaded AI assistant with modular perception system"
readme = "README.md"
authors = [
    { name = "enderdincer", email = "ender95dincer@gmail.com" }
]
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.9.0",
    "numpy>=1.26.0",
    "pillow>=10.2.0",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "soundfile>=0.12.0",
]

[project.optional-dependencies]
# Speech-to-Text support
stt = [
    "librosa>=0.10.0",
]

# Development tools
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-timeout>=2.2.0",
    "pytest-mock>=3.12.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]

# All optional dependencies
all = [
    "ai-assistant[stt,dev]",
]
```

### Alternative: Minimal Core Dependencies
If keeping core lightweight, move ML dependencies to optional:
```toml
dependencies = [
    "opencv-python>=4.9.0",
    "numpy>=1.26.0",
    "pillow>=10.2.0",
]

[project.optional-dependencies]
stt = [
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
]
```

### Installation Commands
Document in README:
```bash
# Core installation
pip install -e .

# With STT support
pip install -e ".[stt]"

# Development environment
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### Platform-Specific Notes

#### macOS (Apple Silicon)
```bash
# For MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio
```

#### Linux (CUDA)
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Validation

### Dependency Resolution
- [ ] Run `pip install -e .` in clean virtual environment
- [ ] Verify no dependency conflicts
- [ ] Check that all imports work

### Version Compatibility
- [ ] Test with Python 3.11
- [ ] Test with Python 3.12 if supported
- [ ] Verify transformers can load models
- [ ] Verify torch can perform inference

### Test Installation
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install package
pip install -e ".[stt,dev]"

# Verify imports
python -c "import transformers; import torch; import torchaudio; print('Success!')"

# Run tests
pytest tests/
```

## Documentation Updates

### README.md
- [ ] Add installation instructions with optional dependencies
- [ ] Document platform-specific installation
- [ ] Add troubleshooting section for common issues

### Example Documentation
```markdown
## Installation

### Basic Installation
```bash
pip install -e .
```

### With Speech-to-Text Support
```bash
pip install -e ".[stt]"
```

### GPU Support
For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Requirements
- Python 3.11+
- For STT: ~4GB disk space for model cache
- For GPU inference: CUDA-compatible GPU with 4GB+ VRAM
```

## Success Metrics
- Dependencies install without conflicts
- All required imports work
- Tests can run with new dependencies
- Documentation is clear and accurate

## Estimated Effort
Small (1-2 hours including testing)

## Related Tasks
- Related: Task 06 (STT Processor - needs these dependencies)
- Next: Task 08 (Integration with PerceptionSystem)
