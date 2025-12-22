# Task 02: Create Base Processor Implementation

## High-Level Objective
Implement an abstract base class that provides common functionality for all sensory processors, including event subscription/unsubscription, lifecycle management, and thread-safe processing.

## Core Direction
Create a reusable foundation that handles boilerplate code for event handling, threading, and lifecycle management. Individual processor implementations should only need to implement the actual processing logic.

## Dependencies
- Task 01: Processor interface must be complete

## Prerequisites
- IProcessor interface defined
- EventBus implementation available
- ThreadManager available

## Acceptance Criteria

### 1. Base Implementation
- [ ] Create `BaseProcessor` abstract class in `src/ai_assistant/perception/processors/base.py`
- [ ] Implement IProcessor interface
- [ ] Provide automatic event subscription on start
- [ ] Provide automatic event unsubscription on stop
- [ ] Handle both sync and async processing patterns
- [ ] Include thread-safe state management

### 2. Abstract Methods
- [ ] Define `_process_event(event: IEvent) -> List[IEvent]` as abstract method
- [ ] Optionally support async variant: `_process_event_async(event: IEvent) -> Awaitable[List[IEvent]]`
- [ ] Subclasses should only implement business logic in these methods

### 3. Error Handling
- [ ] Wrap processing in try-catch to prevent processor crashes from breaking event bus
- [ ] Log errors with appropriate context
- [ ] Continue processing subsequent events after errors
- [ ] Optionally support error event publishing

### 4. Configuration Support
- [ ] Accept configuration dict in constructor
- [ ] Support common processor settings (batch_size, timeout, etc.)
- [ ] Validate configuration on initialization

### 5. Module Structure
- [ ] Create `src/ai_assistant/perception/processors/` directory
- [ ] Create `__init__.py` and export BaseProcessor
- [ ] Update perception module exports

## Implementation Notes

### File Structure
```
src/ai_assistant/perception/
├── processors/                    # NEW directory
│   ├── __init__.py               # NEW: Export BaseProcessor
│   └── base.py                   # NEW: BaseProcessor implementation
└── (existing directories...)
```

### Key Features
1. **Automatic Event Bus Integration**: Subscribe to specified events on start
2. **Lifecycle Management**: Initialize → Start → Stop pattern
3. **Error Isolation**: Catch exceptions to prevent cascade failures
4. **Thread Safety**: Use locks for state changes
5. **Logging**: Rich contextual logging for debugging

### Code Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from ai_assistant.shared.interfaces import IEventBus, IEvent, IProcessor
from ai_assistant.shared.logging import get_logger

class BaseProcessor(IProcessor, ABC):
    """Abstract base class for sensory processors."""
    
    def __init__(
        self,
        processor_id: str,
        processor_type: str,
        event_bus: IEventBus,
        input_event_types: List[str],
        output_event_types: List[str],
        config: Optional[Dict[str, Any]] = None,
    ):
        self._processor_id = processor_id
        self._processor_type = processor_type
        self._event_bus = event_bus
        self._input_event_types = input_event_types
        self._output_event_types = output_event_types
        self._config = config or {}
        self._running = False
        self._initialized = False
        self._logger = get_logger(f"{__name__}.{processor_id}")
    
    def start(self) -> None:
        """Start processor and subscribe to events."""
        # Subscribe to input events
        for event_type in self._input_event_types:
            self._event_bus.subscribe(event_type, self._handle_event)
    
    def stop(self) -> None:
        """Stop processor and unsubscribe from events."""
        # Unsubscribe from input events
        for event_type in self._input_event_types:
            self._event_bus.unsubscribe(event_type, self._handle_event)
    
    def _handle_event(self, event: IEvent) -> None:
        """Internal event handler with error handling."""
        try:
            output_events = self._process_event(event)
            for output_event in output_events:
                self._event_bus.publish(output_event)
        except Exception as e:
            self._logger.error(f"Error processing event: {e}")
    
    @abstractmethod
    def _process_event(self, event: IEvent) -> List[IEvent]:
        """Process event - must be implemented by subclasses."""
        pass
```

## Validation

### Unit Tests
- [ ] Test processor initialization
- [ ] Test event subscription on start
- [ ] Test event unsubscription on stop
- [ ] Test error handling doesn't crash processor
- [ ] Test configuration validation

### Test File Location
- `tests/unit/perception/test_base_processor.py`

## Success Metrics
- BaseProcessor class implemented with all lifecycle methods
- Unit tests pass with >90% coverage
- Can be subclassed for concrete implementations
- Error handling prevents cascade failures

## Estimated Effort
Medium (2-4 hours)

## Related Tasks
- Previous: Task 01 (Processor Interface)
- Next: Task 03 (Processor Manager)
