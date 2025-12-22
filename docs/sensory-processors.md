# Sensory Processor Implementation

This document describes the sensory processor system that allows flexible processing of input sources in the AI Assistant perception system.

## Overview

The sensory processor system enables you to attach multiple processors to a single input source. For example, you can add:
- Speech-to-Text (STT) processor to audio input
- Frequency detection processor to detect specific sounds
- Pattern detection processor for audio events
- Object detection processor for camera frames (future)

## Architecture

```
Audio Input Source
    ↓ publishes "audio.sample" events
Event Bus
    ↓ dispatches to subscribers
STT Processor (subscribes to "audio.sample")
    ↓ processes audio → transcribes
    ↓ publishes "audio.transcription" events
Event Bus
    ↓ dispatches to subscribers
Application Handler (receives transcriptions)
```

## Installation

### Basic Installation (Core Only)

```bash
pip install -e .
```

### With Speech-to-Text Support

```bash
pip install -e ".[stt]"
```

This installs:
- `transformers>=4.40.0` - HuggingFace transformers for model loading
- `torch>=2.2.0` - PyTorch for model inference
- `torchaudio>=2.2.0` - PyTorch audio utilities
- `soundfile>=0.12.0` - Audio file I/O
- `librosa>=0.10.0` - Audio processing utilities

### GPU Support

For CUDA (NVIDIA GPUs):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For Apple Silicon (M1/M2):
```bash
# MPS (Metal Performance Shaders) is automatically supported
pip install torch torchvision torchaudio
```

For CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Development Environment

```bash
pip install -e ".[dev]"
```

## Usage

### Method 1: Configuration-Based

```python
from ai_assistant.perception.core import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.interfaces import IEvent

# Create config with audio input and STT processor
config = PerceptionConfig.with_audio_and_stt(
    audio_source_id="microphone",
    stt_processor_id="stt_en",
    model_name="nvidia/canary-1b",
    device="cpu",  # or "cuda" for GPU
    buffer_duration=3.0,
    language="en",
)

# Create system
system = PerceptionSystem(config)

# Subscribe to transcriptions
def handle_transcription(event: IEvent) -> None:
    text = event.data["text"]
    confidence = event.data["confidence"]
    print(f"Transcribed: {text} (confidence: {confidence:.2f})")

system.subscribe("audio.transcription", handle_transcription)

# Start system
system.start()
```

### Method 2: Dynamic Addition

```python
from ai_assistant.perception.core import PerceptionSystem

# Create system
system = PerceptionSystem()

# Add audio input
system.add_audio_input("microphone", {
    "sample_rate": 16000,
    "chunk_size": 1024,
})

# Add STT processor
system.add_stt_processor("stt_en", {
    "model_name": "nvidia/canary-1b",
    "device": "cpu",
    "buffer_duration": 3.0,
    "language": "en",
    "min_confidence": 0.3,
})

# Subscribe to events
system.subscribe("audio.transcription", handle_transcription)

# Start system
system.start()
```

### Method 3: Multiple Processors on Same Input

```python
# You can attach multiple processors to the same audio input
system.add_audio_input("microphone")

# Add STT processor
system.add_stt_processor("stt_en", {"language": "en"})

# Add other processors (when implemented)
# system.add_frequency_detector("freq_det", {"threshold": 0.5})
# system.add_pattern_detector("pattern_det", {"patterns": [...]})

# Subscribe to different event types
system.subscribe("audio.transcription", handle_transcription)
system.subscribe("audio.frequency.detected", handle_frequency)
system.subscribe("audio.pattern.detected", handle_pattern)
```

## STT Processor Configuration

The STT processor supports the following configuration options:

- `model_name` (str): HuggingFace model identifier
  - Default: `"nvidia/canary-1b"`
  - Alternative: `"nvidia/canary-qwen-2.5b"` (larger, more accurate)
  
- `device` (str): Inference device
  - Options: `"cpu"`, `"cuda"`, `"mps"`
  - Default: `"cpu"`
  
- `buffer_duration` (float): Audio buffer duration in seconds
  - Default: `3.0`
  - Recommendation: 2-5 seconds for good accuracy
  
- `sample_rate` (int): Expected sample rate in Hz
  - Default: `16000`
  - Must match model's expected input
  
- `min_confidence` (float): Minimum confidence threshold (0.0 to 1.0)
  - Default: `0.3`
  - Transcriptions below this threshold are discarded
  
- `language` (str): Language code
  - Default: `"en"`
  - Examples: `"es"`, `"fr"`, `"de"`, etc.

## Events

### AudioTranscriptionEvent

Published by STT processor when audio is transcribed.

**Event Type:** `"audio.transcription"`

**Data Fields:**
```python
{
    "text": str,                    # Transcribed text
    "language": str,                # Language code
    "confidence": float,            # Confidence score (0.0 to 1.0)
    "audio_duration": float,        # Duration of audio in seconds
    "model_name": str,             # Model used for transcription
    "source_event_id": str | None, # ID of source audio event
    "text_length": int,            # Length of transcribed text
    "metadata": dict,              # Additional metadata
}
```

## System Status

Check system status including processors:

```python
status = system.get_status()

print(f"Input Sources: {len(status['input_sources'])}")
print(f"Processors: {len(status['processors']['processors'])}")

# List all processors
processors = system.list_processors()
for proc_id, proc_type in processors.items():
    print(f"  {proc_id}: {proc_type}")
```

## Examples

See the `examples/` directory:
- `stt_example.py` - Basic STT processor usage
- `multi_processor_audio.py` - Multiple processors on same input (future)

## Requirements

### System Requirements
- Python 3.11+
- ~4GB disk space for model cache (first download)
- 4GB+ RAM for model inference
- (Optional) CUDA-compatible GPU with 4GB+ VRAM for GPU acceleration

### Model Requirements
The first time you run the STT processor, it will download the model from HuggingFace:
- `nvidia/canary-1b`: ~1.2GB
- `nvidia/canary-qwen-2.5b`: ~2.5GB

Models are cached in `~/.cache/huggingface/` and only downloaded once.

## Troubleshooting

### Import Errors

If you see import errors for `torch` or `transformers`:
```bash
pip install -e ".[stt]"
```

### CUDA Not Available

If you see "CUDA requested but not available":
1. Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install CUDA-enabled PyTorch (see GPU Support section above)
3. Or use CPU: set `device="cpu"` in processor config

### Model Download Fails

If model download fails:
1. Check internet connection
2. Try manually: `python -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('nvidia/canary-1b')"`
3. Check HuggingFace status: https://status.huggingface.co/

### Out of Memory

If you get OOM errors:
1. Reduce `buffer_duration` to process shorter audio chunks
2. Use smaller model: `nvidia/canary-1b` instead of `nvidia/canary-qwen-2.5b`
3. Use CPU instead of GPU if GPU memory is limited
4. Close other applications to free RAM

## Architecture Details

### Processor Lifecycle

1. **Initialization**: Load model, validate config
2. **Start**: Subscribe to input events
3. **Processing**: 
   - Receive audio events
   - Buffer audio until threshold
   - Run inference
   - Publish transcription events
4. **Stop**: Unsubscribe, cleanup resources

### Thread Safety

All processors are thread-safe:
- Event handling is synchronized
- Buffer access is protected
- State changes use locks

### Error Handling

Processors are designed to fail gracefully:
- Errors are logged but don't crash the system
- Processing continues after errors
- Invalid audio is skipped

## Future Enhancements

Planned processor types:
- Frequency detector for audio
- Pattern detector for audio events
- Object detection for camera frames
- Face detection for camera frames
- Emotion detection from speech

## Contributing

To add a new processor:
1. Extend `BaseProcessor` class
2. Implement `_process_event()` method
3. Define input and output event types
4. Add initialization logic in `_initialize_processor()`
5. Add cleanup logic in `_cleanup_processor()`
6. Export from `processors/__init__.py`

See `STTProcessor` as a reference implementation.
