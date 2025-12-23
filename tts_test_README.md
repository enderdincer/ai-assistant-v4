# TTS Voice Testing Script

Quick guide to test different voices and sentences with the Kokoro TTS module.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -e ".[tts]"
   ```

2. **Download model files:**
   - [kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx) (~300MB)
   - [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) (~1MB)
   
   Place them in the project root directory.

3. **Run the test script:**
   ```bash
   python test_tts_voices.py
   ```

## Features

### Menu Options

**1. Test all voices with first sentence**
- Tests every available voice with a sample sentence
- Great for discovering voices

**2. Test all sentences with one voice**
- Pick a voice and hear all test sentences
- Good for evaluating a specific voice

**3. Test voice groups**
- Test groups like:
  - American Female: af_bella, af_sarah, af_nicole
  - American Male: am_adam, am_michael
  - British Female: bf_emma, bf_isabella
  - British Male: bm_george, bm_lewis

**4. Test speed variations**
- Test a voice at different speeds (0.5x to 2.0x)
- Compare slow, normal, and fast speech

**5. Interactive mode** ⭐ Most useful!
- Type any text and hear it spoken
- Switch voices on the fly
- Adjust speed in real-time

**6. Generate comparison samples**
- Creates samples with popular voices for quick comparison

## Interactive Mode Commands

```
> Hello, this is a test!              # Synthesize text
> v:am_adam                           # Switch to Adam's voice
> s:1.5                               # Set speed to 1.5x
> list                                # Show all available voices
> quit                                # Exit
```

## Example Session

```bash
$ python test_tts_voices.py

Select option (0-6): 5              # Enter interactive mode

> Hello! How are you today?         # Type your text
✓ → interactive_af_bella.wav

> v:am_adam                          # Switch to male voice
✓ Voice changed to: am_adam

> Hello! How are you today?         # Same text, different voice
✓ → interactive_am_adam.wav

> s:1.5                              # Speak faster
✓ Speed changed to: 1.5x

> This is much faster now!
✓ → interactive_am_adam.wav
```

## Output

All generated audio files are saved to `tts_outputs/` directory with descriptive names:
- `af_bella_speed1.0.wav`
- `am_adam_speed1.5.wav`
- `interactive_af_bella.wav`

## Customization

Edit `test_tts_voices.py` to customize:

```python
# Add your own test sentences
TEST_SENTENCES = [
    "Your custom sentence here",
    "Another test sentence",
]

# Change model file paths
MODEL_PATH = Path("path/to/kokoro-v1.0.onnx")
VOICES_PATH = Path("path/to/voices-v1.0.bin")

# Change output directory
OUTPUT_DIR = Path("my_outputs")
```

## Tips

- **Best voices for clarity**: af_bella, am_adam, bf_emma
- **Natural sounding**: af_sarah, bm_george
- **Try different speeds**: 0.75x for clarity, 1.25x for efficiency
- **Compare voices**: Use option 6 to generate quick comparison samples
- **Long texts**: Interactive mode is best for testing paragraphs

## Troubleshooting

**Error: Model files not found**
```bash
# Download the files:
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

**Error: Module not found**
```bash
pip install -e ".[tts]"
```

**Slow performance**
- Try using GPU: Set `device="cuda"` in the config
- Reduce `num_threads` if using CPU
- Use the quantized model (smaller, faster)

## Available Voices

20+ voices including:
- **American Female**: af_bella, af_sarah, af_nicole
- **American Male**: am_adam, am_michael  
- **British Female**: bf_emma, bf_isabella
- **British Male**: bm_george, bm_lewis
- **Japanese**: ja_* voices
- And more...

See full list: [Kokoro Voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
