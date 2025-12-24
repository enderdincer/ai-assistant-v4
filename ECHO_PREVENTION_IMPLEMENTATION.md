# Echo Prevention and Interrupt Detection Implementation

## Summary

Implemented a hybrid approach to prevent echo loops and enable user interruptions in the voice assistant:

1. **Text-based Echo Filtering** - Prevents assistant from responding to its own speech
2. **Energy-based Interrupt Detection** - Allows users to interrupt the assistant while speaking

## Components Implemented

### 1. EchoFilter (`echo_filter.py`)

**Purpose**: Compare transcriptions against recent TTS responses to filter out echo.

**Key Features**:
- Fuzzy string matching (80% similarity threshold)
- Substring matching for partial transcriptions
- Time-windowed filtering (active during TTS + 500ms buffer)
- Stores last 3 responses to handle conversation context
- Case-insensitive matching

**How it works**:
```python
# Store TTS response before speaking
echo_filter.set_tts_response("Hello, how can I help?", duration_ms=2000)

# Check if transcription is echo
if echo_filter.is_echo("Hello, how can I help?"):
    # Discard as echo
```

### 2. Audio Input Source Updates (`src/ai_assistant/perception/input_sources/audio.py`)

**New Features**:
- Energy-based interrupt detection
- Baseline energy calculation (rolling average of 50 chunks)
- Configurable interrupt threshold (default: 3x baseline)
- Interrupt callback mechanism

**Key Methods**:
- `set_interrupt_callback(callback)` - Register handler for interrupts
- `set_tts_state(is_playing)` - Enable/disable interrupt detection
- `_check_interrupt(samples)` - Detect energy spikes during TTS

**How it works**:
```python
# When not speaking, build baseline energy
baseline_energy = rolling_average(recent_audio_chunks)

# During TTS playback
if current_energy > baseline_energy * 3.0:
    # Interrupt detected! Call callback
    interrupt_callback()
```

### 3. TTS Interrupt Support (`src/ai_assistant/actions/tts/kokoro_tts.py`)

**New Features**:
- `stop()` - Immediately stop TTS playback
- `estimate_duration(text)` - Estimate speech duration for echo filtering
- Interruptible `play()` method with stop event checking

**Implementation**:
```python
# Play with interrupt support
self._stop_event.clear()
sd.play(audio, blocking=False)

while sd.get_stream().active:
    if self._stop_event.is_set():
        sd.stop()  # Stop immediately
        break
```

### 4. Assistant Integration (`assistant.py`)

**Updates**:
- Created `EchoFilter` instance on initialization
- Set up interrupt callback on audio source during `start()`
- Updated `_on_transcription()` to:
  - Filter echo before processing
  - Detect interruptions and stop TTS
- Updated `_speak()` to:
  - Store response in echo filter
  - Enable interrupt detection during TTS
  - Disable interrupt detection after TTS

**Flow**:
```
User speaks → STT transcribes → Check echo filter
                                      ↓
                              Not echo? Process
                                      ↓
                              Generate LLM response
                                      ↓
                              Store in echo filter
                                      ↓
                              Enable interrupt detection
                                      ↓
                              Speak via TTS ────→ If loud audio: Stop TTS
                                      ↓
                              Disable interrupt detection
```

## Configuration

### EchoFilter Settings
- `buffer_ms`: 500ms (time after TTS to continue filtering)
- `similarity_threshold`: 0.80 (80% similarity for fuzzy matching)
- `max_stored_responses`: 3 (number of recent responses to keep)

### Audio Interrupt Settings
- `interrupt_threshold`: 2.0 (energy must be 2x TTS baseline)
- `baseline_window`: 50 (chunks for baseline calculation)
- `tts_warmup_chunks`: 10 (chunks to wait before interrupt detection during TTS)
- `interrupt_cooldown_chunks`: 20 (cooldown after interrupt to prevent re-triggers)

## Testing

### Echo Filter Tests
All 10 tests passed:
- ✓ Exact match detection
- ✓ Partial/substring matching
- ✓ Different text rejection
- ✓ Fuzzy matching with typos
- ✓ Time window behavior
- ✓ Multiple stored responses
- ✓ Storage limits
- ✓ Case insensitivity
- ✓ Clear functionality

Run tests with:
```bash
python test_echo_filter.py
```

### Integration Testing

To test the full system:
```bash
python assistant.py --verbose
```

**Expected behavior**:
1. **Echo prevention**: Assistant speech is not transcribed as user input
2. **Interruption**: Speak loudly while assistant is talking → TTS stops immediately
3. **Normal operation**: Regular conversation works without issues

**Look for these logs**:
- `Echo filtered: '<text>'` - Echo successfully filtered
- `🎤 User interrupted assistant` - Interrupt detected
- `Stopping TTS due to interruption` - TTS stopped
- `Interrupt detected! Energy: X, Baseline: Y` - Energy spike details

## How It Works Together

### Scenario 1: Normal Conversation (No Echo)
```
User: "What's the weather?"
  ↓
STT: "What's the weather?"
  ↓
Echo Filter: Not echo (no recent TTS) → Process
  ↓
LLM: "The weather today is sunny."
  ↓
Store in echo filter + Enable interrupt detection
  ↓
TTS plays: "The weather today is sunny."
  ↓
Mic captures TTS audio → STT transcribes: "The weather today is sunny."
  ↓
Echo Filter: Fuzzy match with stored response → DISCARD ✓
  ↓
No echo loop!
```

### Scenario 2: User Interruption
```
TTS playing: "The weather today is..."
  ↓
User speaks loudly: "STOP" or "Wait"
  ↓
Audio source detects energy spike (3x baseline)
  ↓
Calls interrupt_callback()
  ↓
TTS.stop() called → Playback stops immediately
  ↓
STT transcribes user's interruption: "STOP"
  ↓
Echo Filter: Not echo (different from TTS text) → Process
  ↓
Assistant handles interruption
```

### Scenario 3: Delayed Echo (After TTS Ends)
```
TTS finishes at time T
  ↓
Buffer period: T to T+500ms
  ↓
At T+300ms: STT finally processes buffered TTS audio
  ↓
Echo Filter: Within time window + text matches → DISCARD ✓
  ↓
No echo processed!
```

## Limitations

1. **Soft interruptions**: User must speak with enough volume (3x baseline) to trigger interrupt
2. **Acoustic interference**: If TTS is very loud, it may drown out soft user speech
3. **STT accuracy**: Fuzzy matching threshold (80%) may miss very garbled transcriptions
4. **Baseline adaptation**: Takes ~50 chunks to establish stable baseline in new environment

## Future Enhancements

1. **Adaptive thresholds**: Adjust interrupt threshold based on environment
2. **Acoustic Echo Cancellation (AEC)**: Use reference signal for better echo cancellation
3. **Voice activity detection**: Distinguish between human voice and TTS voice
4. **Directional microphones**: Hardware solution to reduce speaker pickup
5. **Headphone mode**: Dedicated mode for headphone users (no echo issues)

## Files Modified

1. **New**: `echo_filter.py` - Text-based echo filtering
2. **Modified**: `src/ai_assistant/perception/input_sources/audio.py` - Interrupt detection
3. **Modified**: `src/ai_assistant/actions/tts/kokoro_tts.py` - Interruptible TTS
4. **Modified**: `assistant.py` - Integration of echo filter and interrupts
5. **New**: `test_echo_filter.py` - Test suite for echo filter

## Usage

The system works automatically once the assistant is started:

```bash
# Start assistant
python assistant.py --verbose

# The assistant will:
# 1. Listen for speech
# 2. Filter out its own speech (echo prevention)
# 3. Allow interruptions (speak loudly during TTS)
# 4. Continue conversation naturally
```

**Interrupting the assistant**:
- Speak with moderate to loud volume while the assistant is talking
- The TTS will stop immediately
- Your interruption will be processed normally

**Tips**:
- For best results, use headphones (prevents echo entirely)
- Speak clearly and with adequate volume for interruptions
- The system adapts to your environment's baseline noise level
