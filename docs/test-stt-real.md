# Real STT Test

This test file (`test_stt_real.py`) tests the STT processor with real audio input and the actual Canary model - **no mocking**.

## Prerequisites

### 1. Install STT Dependencies

```bash
pip install -e ".[stt]"
```

This installs:
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers for model loading
- `torchaudio` - Audio processing
- `soundfile` - Audio file I/O
- `librosa` - Audio utilities

### 2. (Optional) Install PyAudio for Real Microphone

For real microphone input:

```bash
pip install pyaudio
```

**Note:** PyAudio installation can be tricky on some systems:

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
```bash
pip install pyaudio
# Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

If PyAudio installation fails, the test will fall back to simulated audio (which won't produce real transcriptions).

## Running the Test

### Basic Run (CPU)

```bash
python test_stt_real.py
```

### What It Does

1. **Checks Dependencies** - Verifies all required libraries are installed
2. **Loads STT Model** - Downloads nvidia/canary-1b (~1.2GB) on first run
3. **Starts Microphone** - Captures audio from your default microphone
4. **Transcribes Audio** - Every 3 seconds, sends audio to STT model
5. **Displays Results** - Shows transcriptions in real-time

### First Run

The first time you run this, it will download the model:

```
⏳ Please wait while model is loaded...
  (First run will download ~1.2GB model)
```

This takes 2-5 minutes depending on your internet connection. The model is cached in `~/.cache/huggingface/` and only downloaded once.

### Expected Output

```
======================================================================
🎤 REAL STT PROCESSOR TEST
======================================================================

Checking dependencies...
✓ PyTorch version: 2.2.0
  Using CPU (this will be slower)
✓ Transformers version: 4.40.0
✓ Soundfile version: 0.12.1
✓ PyAudio available

======================================================================

🚀 Using GPU acceleration (CUDA)  # or CPU/MPS depending on hardware

📋 Configuring perception system...
🚀 Starting perception system...
✓ System started

📍 Setting up real microphone input...
✓ Real microphone configured
  Sample rate: 16000 Hz
  Channels: 1 (mono)
  Chunk size: 1024 samples

🤖 Loading STT model...
  Model: nvidia/canary-1b
  Device: cpu
  Language: English

⏳ Please wait while model is loaded...
  (First run will download ~1.2GB model)
✓ STT model loaded in 45.2s

======================================================================
📊 SYSTEM STATUS
======================================================================
Input Sources:  ['microphone']
Processors:     ['stt_en']
Event Bus:      Running
Queue Size:     0
======================================================================

======================================================================
🎙️  READY TO TRANSCRIBE
======================================================================

The system is now listening to your microphone!
Audio will be transcribed every 3 seconds.

Instructions:
  • Speak clearly into your microphone
  • Wait for transcription results to appear
  • Press Ctrl+C to stop

Example phrases to try:
  • 'Hello, this is a test'
  • 'The weather is nice today'
  • 'Can you hear me clearly?'

======================================================================

# When you speak, you'll see:

======================================================================
🎙️  NEW TRANSCRIPTION
======================================================================
Text:       Hello, this is a test
Language:   en
Confidence: 0.987
Duration:   2.85s
Model:      nvidia/canary-1b
======================================================================
```

## Configuration

You can modify the STT processor configuration in the test file:

```python
system.add_stt_processor("stt_en", {
    "model_name": "nvidia/canary-1b",     # or "nvidia/canary-qwen-2.5b"
    "device": "cpu",                       # or "cuda", "mps"
    "buffer_duration": 3.0,                # seconds of audio to buffer
    "sample_rate": 16000,                  # Hz
    "language": "en",                      # language code
    "min_confidence": 0.01,                # minimum confidence threshold
})
```

### Model Options

- `nvidia/canary-1b` - Smaller, faster (~1.2GB, good for testing)
- `nvidia/canary-qwen-2.5b` - Larger, more accurate (~2.5GB, better quality)

### Device Options

- `"cpu"` - Works everywhere, slower
- `"cuda"` - NVIDIA GPU, much faster (requires CUDA)
- `"mps"` - Apple Silicon GPU, faster on M1/M2/M3 Macs

## Troubleshooting

### No Transcriptions

If you don't get any transcriptions:

1. **Check microphone permissions** - Make sure Python has microphone access
2. **Speak louder** - Audio might be too quiet
3. **Check audio input** - Test your microphone in system settings
4. **Reduce buffer duration** - Try 2.0 seconds instead of 3.0

### "CUDA out of memory"

If you get OOM errors on GPU:

1. Use smaller model: `nvidia/canary-1b`
2. Switch to CPU: `device="cpu"`
3. Close other GPU-using applications

### "PyAudio not available"

If PyAudio installation fails:

1. The test will still run with simulated audio
2. For real audio, follow the PyAudio installation instructions above
3. Or use a different audio library (requires modifying the audio input source)

### Model Download Fails

If model download fails:

1. Check internet connection
2. Check HuggingFace status: https://status.huggingface.co/
3. Try again - downloads resume automatically
4. Manual download:
   ```python
   from transformers import AutoModelForSpeechSeq2Seq
   AutoModelForSpeechSeq2Seq.from_pretrained("nvidia/canary-1b")
   ```

## System Requirements

- **Python:** 3.11+
- **RAM:** 4GB+ (8GB recommended)
- **Disk:** 4GB free space (for model cache)
- **CPU:** Any modern CPU works (faster is better)
- **GPU (optional):** NVIDIA GPU with 4GB+ VRAM or Apple Silicon M1/M2/M3

## Performance

Expected transcription latency:

- **CPU:** 2-5 seconds per 3-second audio clip
- **GPU (CUDA):** 0.3-1 second per 3-second audio clip  
- **MPS (Apple):** 0.5-1.5 seconds per 3-second audio clip

## What's Being Tested

This test validates:

1. ✅ **Real audio capture** - Microphone input works
2. ✅ **Audio buffering** - Audio chunks are accumulated properly
3. ✅ **Model loading** - Canary model loads successfully
4. ✅ **Inference** - Model can transcribe audio
5. ✅ **Event flow** - Events flow through the system correctly
6. ✅ **Processor lifecycle** - Processor starts/stops cleanly
7. ✅ **Real-time processing** - System handles continuous audio

No mocks, no stubs - everything is production code!
