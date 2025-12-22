# Task 05: Create Audio Utilities Module

## High-Level Objective
Implement utility functions for audio processing including audio buffering, resampling, format conversion, and Voice Activity Detection (VAD) preparation for the STT processor.

## Core Direction
Create reusable audio processing utilities that handle the complexities of audio manipulation, allowing the STT processor to focus on transcription logic. These utilities should be general-purpose and reusable for future audio processors.

## Dependencies
- None (can be done in parallel with other tasks)

## Prerequisites
- NumPy available in dependencies
- Understanding of audio processing concepts (sample rates, channels, etc.)

## Acceptance Criteria

### 1. Audio Buffer Management
- [ ] Create `AudioBuffer` class for accumulating audio chunks
- [ ] Support configurable buffer size (in samples or duration)
- [ ] Provide `add_chunk(samples)` method to append audio
- [ ] Provide `get_buffer()` method to retrieve accumulated audio
- [ ] Provide `clear()` method to reset buffer
- [ ] Support overlap/sliding window if needed

### 2. Audio Format Conversion
- [ ] Implement `resample_audio(samples, original_rate, target_rate)` function
- [ ] Implement `convert_to_mono(samples, channels)` if multi-channel
- [ ] Implement `normalize_audio(samples)` for volume normalization
- [ ] Implement `float_to_int16(samples)` and `int16_to_float(samples)` conversions
- [ ] Handle edge cases (empty arrays, invalid rates, etc.)

### 3. Voice Activity Detection (VAD) Support
- [ ] Implement `compute_energy(samples)` for simple energy-based VAD
- [ ] Implement `is_speech(samples, threshold)` helper
- [ ] Optionally integrate with webrtcvad or silero-vad if available
- [ ] Provide configuration for VAD sensitivity

### 4. Utility Functions
- [ ] `compute_duration(samples, sample_rate)` - Calculate audio duration
- [ ] `trim_silence(samples, threshold)` - Remove leading/trailing silence
- [ ] `split_on_silence(samples, sample_rate, min_silence_duration)` - Split audio chunks

### 5. Module Structure
- [ ] Create `src/ai_assistant/perception/processors/audio_utils.py`
- [ ] Add comprehensive docstrings
- [ ] Include type hints for all functions
- [ ] Export utilities from processors module

## Implementation Notes

### File Structure
```
src/ai_assistant/perception/processors/
├── __init__.py          # Export utilities
├── base.py             # BaseProcessor
├── manager.py          # ProcessorManager
└── audio_utils.py      # NEW: Audio utilities
```

### Key Components

#### AudioBuffer Class
```python
class AudioBuffer:
    """Buffer for accumulating audio chunks."""
    
    def __init__(self, max_duration: float, sample_rate: int):
        """Initialize buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self._max_samples = int(max_duration * sample_rate)
        self._buffer = np.array([], dtype=np.float32)
        self._sample_rate = sample_rate
    
    def add_chunk(self, samples: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        self._buffer = np.concatenate([self._buffer, samples])
        if len(self._buffer) > self._max_samples:
            # Keep most recent samples
            self._buffer = self._buffer[-self._max_samples:]
    
    def get_buffer(self) -> np.ndarray:
        """Get current buffer contents."""
        return self._buffer.copy()
    
    def clear(self) -> None:
        """Clear buffer."""
        self._buffer = np.array([], dtype=np.float32)
    
    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return len(self._buffer) / self._sample_rate
```

#### Resampling Function
```python
def resample_audio(
    samples: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        samples: Input audio samples
        original_rate: Original sample rate in Hz
        target_rate: Target sample rate in Hz
    
    Returns:
        Resampled audio array
    """
    if original_rate == target_rate:
        return samples
    
    # Use scipy.signal.resample or simple linear interpolation
    # For MVP, can use basic linear interpolation
    duration = len(samples) / original_rate
    target_length = int(duration * target_rate)
    return np.interp(
        np.linspace(0, len(samples), target_length),
        np.arange(len(samples)),
        samples,
    )
```

### Design Considerations
- **Numpy-based**: Use NumPy for efficient array operations
- **Type Safety**: Proper type hints for arrays (use npt.NDArray)
- **Error Handling**: Validate inputs (sample rates, array shapes)
- **Memory Efficiency**: Avoid unnecessary copies
- **Extensibility**: Easy to add more utilities later

### Optional Enhancements (Future)
- Integration with librosa for advanced audio processing
- Spectrogram generation
- Advanced VAD using pre-trained models
- Audio format I/O (WAV, MP3)

## Validation

### Unit Tests
- [ ] Test AudioBuffer accumulation and overflow handling
- [ ] Test resampling with various rates
- [ ] Test format conversions (float/int16)
- [ ] Test normalization
- [ ] Test energy computation and VAD
- [ ] Test edge cases (empty arrays, zero sample rates)

### Test File Location
- `tests/unit/perception/test_audio_utils.py`

## Success Metrics
- All utility functions implemented and tested
- Unit tests pass with >90% coverage
- Functions handle edge cases gracefully
- Ready for integration with STT processor

## Estimated Effort
Medium (3-4 hours)

## Related Tasks
- Previous: Task 04 (Audio Transcription Event)
- Next: Task 06 (STT Processor Implementation)
