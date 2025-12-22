# Task 06: Implement Speech-to-Text Processor

## High-Level Objective
Create a concrete STT processor that subscribes to AudioSampleEvents, buffers audio, performs speech-to-text transcription using the nvidia/canary-qwen-2.5b model, and publishes AudioTranscriptionEvents.

## Core Direction
This is the first concrete processor implementation demonstrating the full processor pattern. It should be production-ready for the MVP and serve as a reference for future processors.

## Dependencies
- Task 02: BaseProcessor must be complete
- Task 04: AudioTranscriptionEvent must be defined
- Task 05: Audio utilities must be available

## Prerequisites
- transformers library available
- torch available for model inference
- Audio utilities implemented
- Event types defined

## Acceptance Criteria

### 1. Processor Implementation
- [ ] Create `STTProcessor` class in `src/ai_assistant/perception/processors/stt_processor.py`
- [ ] Extend BaseProcessor
- [ ] Subscribe to `"audio.sample"` events
- [ ] Publish `"audio.transcription"` events
- [ ] Implement buffering strategy for audio accumulation

### 2. Model Integration
- [ ] Load nvidia/canary-qwen-2.5b model from HuggingFace
- [ ] Initialize model on specified device (CPU/CUDA)
- [ ] Handle model loading errors gracefully
- [ ] Support model caching to avoid re-downloads
- [ ] Implement proper tokenization and inference

### 3. Audio Processing Pipeline
- [ ] Buffer incoming AudioSampleEvents using AudioBuffer
- [ ] Accumulate audio until sufficient duration (e.g., 3-5 seconds)
- [ ] Optional: Implement VAD to detect speech segments
- [ ] Resample audio to model's expected sample rate (16kHz typical)
- [ ] Convert audio to required format (mono, float32)

### 4. Transcription Logic
- [ ] Process buffered audio through model
- [ ] Extract transcription text from model output
- [ ] Calculate confidence score if available
- [ ] Detect language if model supports it
- [ ] Handle empty/silence audio (don't publish empty transcriptions)

### 5. Configuration Support
- [ ] `model_name`: HuggingFace model identifier (default: "nvidia/canary-1b")
- [ ] `device`: Inference device ("cpu", "cuda", "mps")
- [ ] `buffer_duration`: Audio buffer duration in seconds (default: 3.0)
- [ ] `sample_rate`: Expected sample rate (default: 16000)
- [ ] `min_confidence`: Minimum confidence threshold (default: 0.3)
- [ ] `language`: Fixed language or "auto" for detection

### 6. Error Handling
- [ ] Handle model loading failures
- [ ] Handle inference errors
- [ ] Handle audio format issues
- [ ] Log errors without crashing
- [ ] Continue processing after errors

### 7. Performance Optimization
- [ ] Load model once during initialization
- [ ] Reuse model across multiple transcriptions
- [ ] Consider batching if multiple audio chunks ready
- [ ] Monitor inference latency and log warnings if slow

## Implementation Notes

### File Structure
```
src/ai_assistant/perception/processors/
├── __init__.py            # Export STTProcessor
├── base.py               # BaseProcessor
├── manager.py            # ProcessorManager
├── audio_utils.py        # Audio utilities
└── stt_processor.py      # NEW: STT processor
```

### Code Pattern
```python
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from ai_assistant.perception.processors.base import BaseProcessor
from ai_assistant.perception.processors.audio_utils import AudioBuffer, resample_audio
from ai_assistant.shared.interfaces import IEventBus, IEvent
from ai_assistant.shared.events import AudioSampleEvent, AudioTranscriptionEvent
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)

class STTProcessor(BaseProcessor):
    """Speech-to-Text processor using HuggingFace Transformers."""
    
    def __init__(
        self,
        processor_id: str,
        event_bus: IEventBus,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        
        super().__init__(
            processor_id=processor_id,
            processor_type="stt",
            event_bus=event_bus,
            input_event_types=["audio.sample"],
            output_event_types=["audio.transcription"],
            config=config,
        )
        
        # Configuration
        self._model_name = config.get("model_name", "nvidia/canary-1b")
        self._device = config.get("device", "cpu")
        self._buffer_duration = config.get("buffer_duration", 3.0)
        self._target_sample_rate = config.get("sample_rate", 16000)
        self._min_confidence = config.get("min_confidence", 0.3)
        self._language = config.get("language", "en")
        
        # Audio buffer
        self._audio_buffer = AudioBuffer(
            max_duration=self._buffer_duration,
            sample_rate=self._target_sample_rate,
        )
        
        # Model (loaded during initialization)
        self._model = None
        self._processor = None
    
    def _initialize_source(self) -> None:
        """Initialize and load the STT model."""
        logger.info(f"Loading STT model: {self._model_name}")
        
        try:
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            
            logger.info(f"STT model loaded on device: {self._device}")
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise
    
    def _cleanup_source(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        logger.info("STT model unloaded")
    
    def _process_event(self, event: IEvent) -> List[IEvent]:
        """Process audio sample event."""
        if event.event_type != "audio.sample":
            return []
        
        # Extract audio data
        audio_data = event.data
        samples = audio_data["samples"]
        sample_rate = audio_data["sample_rate"]
        
        # Resample if necessary
        if sample_rate != self._target_sample_rate:
            samples = resample_audio(samples, sample_rate, self._target_sample_rate)
        
        # Add to buffer
        self._audio_buffer.add_chunk(samples)
        
        # Check if buffer has enough audio
        if self._audio_buffer.duration < self._buffer_duration:
            return []  # Not enough audio yet
        
        # Transcribe buffered audio
        buffered_audio = self._audio_buffer.get_buffer()
        self._audio_buffer.clear()
        
        try:
            transcription = self._transcribe(buffered_audio)
            
            if transcription and transcription["text"].strip():
                # Create transcription event
                transcription_event = AudioTranscriptionEvent.create(
                    source=self._processor_id,
                    text=transcription["text"],
                    language=transcription["language"],
                    confidence=transcription["confidence"],
                    audio_duration=len(buffered_audio) / self._target_sample_rate,
                    model_name=self._model_name,
                    source_event_id=str(id(event)),
                )
                return [transcription_event]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
        
        return []
    
    def _transcribe(self, audio: np.ndarray) -> Optional[Dict[str, Any]]:
        """Transcribe audio using the model."""
        # Prepare inputs
        inputs = self._processor(
            audio,
            sampling_rate=self._target_sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs)
        
        # Decode transcription
        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        return {
            "text": transcription,
            "language": self._language,
            "confidence": 1.0,  # Model may not provide confidence
        }
```

### Design Considerations
- **Buffering Strategy**: Accumulate 3-5 seconds before transcribing for better accuracy
- **Model Selection**: nvidia/canary-1b is smaller and faster than 2.5b, good for MVP
- **Device Flexibility**: Support CPU/GPU for different deployment scenarios
- **Lazy Loading**: Load model during initialize(), not constructor
- **Memory Management**: Clear buffer after transcription to prevent memory buildup

## Validation

### Unit Tests
- [ ] Test processor initialization
- [ ] Test audio buffering
- [ ] Test resampling integration
- [ ] Mock model inference and test transcription event creation
- [ ] Test error handling

### Integration Tests
- [ ] Test with real audio samples
- [ ] Test end-to-end with PerceptionSystem
- [ ] Verify transcription quality with known audio

### Test File Locations
- `tests/unit/perception/test_stt_processor.py`
- `tests/integration/perception/test_stt_integration.py`

## Success Metrics
- STTProcessor successfully transcribes audio
- Publishes AudioTranscriptionEvents correctly
- Handles errors without crashing
- Tests pass with >80% coverage
- Transcription latency acceptable (< 1 second for 3 second audio on GPU)

## Estimated Effort
Large (6-8 hours)

## Related Tasks
- Previous: Task 05 (Audio Utilities)
- Next: Task 07 (Update Dependencies)
- Related: Task 09 (System Integration)
