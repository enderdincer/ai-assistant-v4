# Task 04: Create Audio Transcription Event

## High-Level Objective
Define a specialized event type for audio transcription results that will be published by the Speech-to-Text processor.

## Core Direction
Follow existing event patterns (CameraFrameEvent, TextInputEvent, AudioSampleEvent) to create a consistent event type for transcribed audio. This event represents the output of STT processing.

## Dependencies
- None (can be done in parallel with processor infrastructure)

## Prerequisites
- Understanding of existing specialized events in `src/ai_assistant/shared/events/specialized.py`
- Event and EventPriority interfaces

## Acceptance Criteria

### 1. Event Definition
- [ ] Add `AudioTranscriptionEvent` to `src/ai_assistant/shared/events/specialized.py`
- [ ] Follow dataclass frozen pattern like other specialized events
- [ ] Include factory method: `create()` for constructing events

### 2. Event Data Structure
Event must contain:
- [ ] `text: str` - The transcribed text
- [ ] `language: str` - Detected/specified language code
- [ ] `confidence: float` - Confidence score (0.0 to 1.0)
- [ ] `audio_duration: float` - Duration of audio that was transcribed (seconds)
- [ ] `source_event_id: Optional[str]` - ID of original AudioSampleEvent (for tracing)
- [ ] `model_name: str` - Name of STT model used
- [ ] Any additional metadata in nested dict

### 3. Event Type
- [ ] Use event_type: `"audio.transcription"`
- [ ] Default priority: `EventPriority.HIGH` (transcribed user input is important)

### 4. Export
- [ ] Export AudioTranscriptionEvent from `src/ai_assistant/shared/events/__init__.py`

## Implementation Notes

### File Location
- Modify: `src/ai_assistant/shared/events/specialized.py`
- Modify: `src/ai_assistant/shared/events/__init__.py`

### Code Pattern
```python
@dataclass(frozen=True)
class AudioTranscriptionEvent(Event):
    """Event containing audio transcription result."""

    @staticmethod
    def create(
        source: str,
        text: str,
        language: str,
        confidence: float,
        audio_duration: float,
        model_name: str,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.HIGH,
    ) -> "AudioTranscriptionEvent":
        """Create an audio transcription event.

        Args:
            source: Processor identifier that created this transcription
            text: The transcribed text
            language: Language code (e.g., 'en', 'es')
            confidence: Confidence score (0.0 to 1.0)
            audio_duration: Duration of audio in seconds
            model_name: Name of the STT model used
            source_event_id: Optional ID of source AudioSampleEvent
            metadata: Optional additional metadata
            priority: Event priority (default HIGH for user input)

        Returns:
            AudioTranscriptionEvent: The created event
        """
        data = {
            "text": text,
            "language": language,
            "confidence": confidence,
            "audio_duration": audio_duration,
            "model_name": model_name,
            "source_event_id": source_event_id,
            "text_length": len(text),
            "metadata": metadata or {},
        }
        return AudioTranscriptionEvent(
            event_type="audio.transcription",
            source=source,
            data=data,
            priority=priority,
        )
```

### Design Considerations
- **Confidence Score**: Allows downstream consumers to filter low-confidence transcriptions
- **Language Detection**: Supports multilingual scenarios
- **Duration Tracking**: Helps with performance monitoring
- **Source Tracing**: Links processed event back to raw event for debugging
- **Model Tracking**: Important for A/B testing different models

## Validation

### Unit Tests
- [ ] Test event creation with all parameters
- [ ] Test event creation with minimal parameters (defaults)
- [ ] Test event immutability (frozen dataclass)
- [ ] Test event serialization (to_dict)
- [ ] Test event ordering (priority comparison)

### Test File Location
- Add tests to existing: `tests/unit/shared/test_specialized_events.py` (or create if doesn't exist)

## Success Metrics
- Event class defined and exported
- Follows existing patterns and conventions
- Unit tests pass
- Ready for use by STT processor

## Estimated Effort
Small (< 1 hour)

## Related Tasks
- Previous: Task 03 (Processor Manager)
- Next: Task 05 (Audio Utilities)
- Related: Task 06 (STT Processor Implementation)
