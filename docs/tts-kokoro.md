# Kokoro TTS Module

This module provides Text-to-Speech (TTS) capabilities using the Kokoro 82M model via ONNX Runtime with **direct audio playback support**.

## Features

- 🎯 **High-quality speech synthesis** using Kokoro 82M model
- 🚀 **Fast inference** with ONNX Runtime (CPU and GPU support)
- 🔊 **Direct audio playback** - plays audio immediately without saving files
- 🗣️ **Multiple voices** (54+ voices including male and female variants)
- 🌍 **Multi-language support** (English, Japanese, Chinese, and more)
- ⚙️ **Configurable** speech speed and voice selection
- 💾 **Optional audio export** to WAV, FLAC, and other formats

## Installation

Install the TTS dependencies:

```bash
pip install -e ".[tts]"
```

This will install:
- `kokoro-onnx` - Kokoro TTS library
- `onnxruntime` - ONNX Runtime for inference
- `soundfile` - Audio file I/O

## Model Files

Download the required model files from the [Kokoro-ONNX releases](https://github.com/thewh1teagle/kokoro-onnx/releases):

1. **kokoro-v1.0.onnx** (~300MB) - The main model file
2. **voices-v1.0.bin** (~1MB) - Voice embeddings

Place these files in your working directory or specify custom paths in the configuration.

## Quick Start

```python
from pathlib import Path
from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig

# Configure TTS
config = KokoroTTSConfig(
    model_path="kokoro-v1.0.onnx",
    voices_path="voices-v1.0.bin",
    voice="af_bella",  # Female voice
    speed=1.0,
)

# Initialize and synthesize
tts = KokoroTTS(config)
tts.initialize()

audio = tts.synthesize("Hello, world!")
tts.save_to_file(audio, "output.wav")

tts.cleanup()
```

## Configuration

### KokoroTTSConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str/Path | Required | Path to kokoro-v1.0.onnx |
| `voices_path` | str/Path | Required | Path to voices-v1.0.bin |
| `voice` | str | "af_bella" | Voice name to use |
| `speed` | float | 1.0 | Speech speed multiplier |
| `lang` | str | "en-us" | Language code |
| `sample_rate` | int | 24000 | Audio sample rate (Hz) |
| `device` | str | "cpu" | Device: "cpu" or "cuda" |
| `num_threads` | int | None | CPU threads (None = auto) |

## Available Voices

The Kokoro model supports 20+ voices across different languages:

### English Voices
- `af_bella` - Female (default)
- `af_sarah` - Female
- `af_nicole` - Female
- `am_adam` - Male
- `am_michael` - Male
- `bf_emma` - British Female
- `bf_isabella` - British Female
- `bm_george` - British Male
- `bm_lewis` - British Male

### Other Languages
- Japanese voices: `ja_*`
- And more...

See [Kokoro-82M/VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) for the complete list.

## Usage Examples

### Basic Usage

```python
from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig

config = KokoroTTSConfig(
    model_path="kokoro-v1.0.onnx",
    voices_path="voices-v1.0.bin",
)

tts = KokoroTTS(config)
tts.initialize()

# Synthesize speech
audio = tts.synthesize("Hello, this is a test.")
tts.save_to_file(audio, "output.wav")
```

### Using Different Voices

```python
# Initialize with default voice
tts = KokoroTTS(config)
tts.initialize()

# Synthesize with different voices
audio1 = tts.synthesize("Speaking with Bella's voice.", voice="af_bella")
audio2 = tts.synthesize("Speaking with Adam's voice.", voice="am_adam")

tts.save_to_file(audio1, "bella.wav")
tts.save_to_file(audio2, "adam.wav")
```

### Context Manager Pattern

```python
# Automatic initialization and cleanup
with KokoroTTS(config) as tts:
    audio = tts.synthesize("Using context manager!")
    tts.save_to_file(audio, "output.wav")
# Cleanup happens automatically
```

### Adjusting Speed

```python
config = KokoroTTSConfig(
    model_path="kokoro-v1.0.onnx",
    voices_path="voices-v1.0.bin",
    speed=1.5,  # 1.5x faster
)

tts = KokoroTTS(config)
tts.initialize()
audio = tts.synthesize("This will be spoken faster.")
```

### GPU Acceleration

```python
config = KokoroTTSConfig(
    model_path="kokoro-v1.0.onnx",
    voices_path="voices-v1.0.bin",
    device="cuda",  # Use GPU
)

tts = KokoroTTS(config)
tts.initialize()
audio = tts.synthesize("Using GPU for faster inference!")
```

### List Available Voices

```python
tts = KokoroTTS(config)
tts.initialize()

voices = tts.get_available_voices()
print(f"Available voices: {', '.join(voices)}")
```

## Running the Example

See `examples/tts_example.py` for a complete example:

```bash
# Download model files first
cd examples
# Then run the example
python tts_example.py
```

## Architecture

The module consists of:

1. **KokoroTTSConfig** - Configuration dataclass for TTS settings
2. **KokoroTTS** - Main TTS engine class
   - Model loading and initialization
   - Voice management
   - Speech synthesis
   - Audio file export

## Performance

- **CPU**: Near real-time on modern CPUs (M1, Intel i7+)
- **GPU**: Significantly faster with CUDA support
- **Model Size**: ~300MB (quantized version available: ~80MB)
- **Sample Rate**: 24kHz (high quality)

## Error Handling

The module provides clear error messages for common issues:

```python
try:
    tts = KokoroTTS(config)
    tts.initialize()
except FileNotFoundError:
    print("Model files not found")
except ImportError:
    print("Required dependencies not installed")
except RuntimeError as e:
    print(f"Initialization failed: {e}")
```

## Dependencies

- **kokoro-onnx** - Kokoro TTS implementation
- **onnxruntime** - ONNX model inference
- **soundfile** - Audio file I/O
- **numpy** - Array operations

## License

- This module: MIT License
- Kokoro model: Apache 2.0

## References

- [Kokoro-ONNX GitHub](https://github.com/thewh1teagle/kokoro-onnx)
- [Kokoro-TTS HuggingFace](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- [Model Files (Releases)](https://github.com/thewh1teagle/kokoro-onnx/releases)
