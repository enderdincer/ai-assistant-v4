# Voice Assistant Quick Start Guide

This guide will help you quickly get the voice-driven conversational assistant running.

## Prerequisites

1. **Ollama** with a model installed:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull a model (qwen3:0.6b is fast and efficient for voice)
   ollama pull qwen3:0.6b
   ```

2. **Audio dependencies** (macOS):
   ```bash
   brew install portaudio
   ```

## Installation

```bash
# Install with STT and TTS support
pip install -e ".[stt,tts]"
```

The TTS models are already downloaded in `.downloaded_models/`:
- `kokoro-v1.0.onnx` (310MB) - Kokoro 82M TTS model
- `voices-v1.0.bin` (27MB) - Voice vectors for 26 voices

## Running the Assistant

```bash
# Basic usage - just run it!
python assistant.py

# Speak into your microphone and the assistant will respond
```

### Custom Configuration

```bash
# Use a different LLM model
python assistant.py --model llama3:8b

# Use a different voice
python assistant.py --voice af_bella

# Adjust speech speed
python assistant.py --speed 1.1

# Verbose mode for debugging
python assistant.py --verbose

# List all available voices
python assistant.py --list-voices
```

### Environment Variables

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen3:0.6b"
export TTS_VOICE="af_heart"

python assistant.py
```

## How It Works

```
Audio Input → STT → Transcription Event → LLM → Response → TTS → Audio Output
```

1. **Audio Input**: Microphone captures your voice
2. **STT**: Speech-to-text with Voice Activity Detection transcribes
3. **Event**: `audio.transcription` event published
4. **LLM**: Ollama processes with conversation history
5. **TTS**: Kokoro synthesizes response
6. **Output**: Response played through speakers

## Available Voices

### American Female (af_*)
- `af_heart` (default), `af_bella`, `af_sarah`, `af_nicole`, `af_sky`, and more

### American Male (am_*)
- `am_adam`, `am_eric`, `am_liam`, `am_michael`, and more

### British Female (bf_*)
- `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`

### British Male (bm_*)
- `bm_daniel`, `bm_george`, `bm_lewis`, `bm_fable`

## Troubleshooting

### Microphone not working
```bash
# macOS: Grant microphone permissions
# System Settings > Privacy & Security > Microphone > Terminal

# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Ollama not responding
```bash
# Check if Ollama is running
ollama list

# Test connection
curl http://localhost:11434/api/tags
```

### Model files missing
The models should already be in `.downloaded_models/`. If not:
```bash
cd .downloaded_models
# Download kokoro-v1.0.onnx (310MB)
curl -L -o kokoro-v1.0.onnx \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
# Download voices-v1.0.bin (27MB)
curl -L -o voices-v1.0.bin \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

## Example Conversation

```bash
$ python assistant.py

[2025-12-23 22:30:00] INFO: AI Assistant started - listening for speech...
[2025-12-23 22:30:05] INFO: User: Hello, how are you? (confidence: 0.95)
[2025-12-23 22:30:06] INFO: Assistant: I'm doing great! How can I help you today?
[2025-12-23 22:30:12] INFO: User: Tell me about Python. (confidence: 0.93)
[2025-12-23 22:30:14] INFO: Assistant: Python is a versatile programming language known for its simplicity and readability. It's great for beginners and widely used in web development, data science, and automation!
```

## For More Information

- See `README.md` for full project documentation
- See `examples/` for more usage examples
- See `src/ai_assistant/` for API documentation
