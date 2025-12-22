# Task 03: Create Processor Manager

## High-Level Objective
Implement a ProcessorManager class to handle registration, lifecycle management, and coordination of multiple processors within the perception system.

## Core Direction
Provide a centralized manager for processors similar to how ThreadManager manages threads. This manager will handle processor registration, initialization, starting, stopping, and provide visibility into active processors.

## Dependencies
- Task 02: BaseProcessor must be complete

## Prerequisites
- BaseProcessor implementation available
- IProcessor interface defined
- ThreadManager pattern understood (for consistency)

## Acceptance Criteria

### 1. Manager Implementation
- [ ] Create `ProcessorManager` class in `src/ai_assistant/perception/processors/manager.py`
- [ ] Support processor registration with unique IDs
- [ ] Support processor unregistration
- [ ] Provide bulk operations (initialize_all, start_all, stop_all)
- [ ] Prevent duplicate processor IDs

### 2. Lifecycle Management
- [ ] `register_processor(processor: IProcessor) -> None` - Register a processor
- [ ] `unregister_processor(processor_id: str) -> None` - Remove a processor
- [ ] `initialize_all() -> None` - Initialize all registered processors
- [ ] `start_all() -> None` - Start all registered processors
- [ ] `stop_all() -> None` - Stop all registered processors in reverse order
- [ ] Support individual processor control: `start_processor(processor_id)`, `stop_processor(processor_id)`

### 3. Query Methods
- [ ] `get_processor(processor_id: str) -> Optional[IProcessor]` - Get processor by ID
- [ ] `list_processors() -> Dict[str, str]` - List processor IDs and types
- [ ] `get_status() -> Dict[str, Any]` - Get status of all processors
- [ ] `is_running() -> bool` - Check if manager has running processors

### 4. Error Handling
- [ ] Handle processor initialization failures gracefully
- [ ] Log errors but continue with other processors
- [ ] Provide detailed error context

### 5. Thread Safety
- [ ] Use locks for registration/unregistration operations
- [ ] Ensure thread-safe access to processor registry

## Implementation Notes

### File Structure
```
src/ai_assistant/perception/processors/
├── __init__.py          # Export BaseProcessor, ProcessorManager
├── base.py             # BaseProcessor (from Task 02)
└── manager.py          # NEW: ProcessorManager
```

### Key Features
1. **Registry Pattern**: Maintain dict of processor_id -> IProcessor
2. **Bulk Operations**: Initialize/start/stop all processors at once
3. **Graceful Degradation**: One processor failure doesn't stop others
4. **Status Reporting**: Visibility into all processor states
5. **Lifecycle Ordering**: Stop in reverse of start order for cleanup

### Code Pattern
```python
from typing import Dict, Optional, Any
from threading import Lock
from ai_assistant.shared.interfaces import IProcessor
from ai_assistant.shared.logging import get_logger

class ProcessorManager:
    """Manages lifecycle of sensory processors."""
    
    def __init__(self) -> None:
        self._processors: Dict[str, IProcessor] = {}
        self._lock = Lock()
        self._logger = get_logger(__name__)
    
    def register_processor(self, processor: IProcessor) -> None:
        """Register a processor."""
        with self._lock:
            processor_id = processor.processor_id
            if processor_id in self._processors:
                raise ValueError(f"Processor '{processor_id}' already registered")
            self._processors[processor_id] = processor
            self._logger.info(f"Registered processor: {processor_id}")
    
    def start_all(self) -> None:
        """Start all registered processors."""
        for processor_id, processor in self._processors.items():
            try:
                processor.start()
                self._logger.info(f"Started processor: {processor_id}")
            except Exception as e:
                self._logger.error(f"Failed to start processor {processor_id}: {e}")
    
    def stop_all(self) -> None:
        """Stop all processors in reverse order."""
        for processor_id, processor in reversed(list(self._processors.items())):
            try:
                processor.stop()
                self._logger.info(f"Stopped processor: {processor_id}")
            except Exception as e:
                self._logger.error(f"Failed to stop processor {processor_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all processors."""
        return {
            "processor_count": len(self._processors),
            "processors": {
                pid: {
                    "type": p.processor_type,
                    "running": p.is_running(),
                }
                for pid, p in self._processors.items()
            }
        }
```

## Validation

### Unit Tests
- [ ] Test processor registration
- [ ] Test duplicate ID rejection
- [ ] Test bulk operations (start_all, stop_all)
- [ ] Test individual processor control
- [ ] Test status reporting
- [ ] Test error handling doesn't cascade

### Test File Location
- `tests/unit/perception/test_processor_manager.py`

## Success Metrics
- ProcessorManager handles multiple processors correctly
- Thread-safe operations
- Graceful error handling
- Unit tests pass with >90% coverage

## Estimated Effort
Medium (2-4 hours)

## Related Tasks
- Previous: Task 02 (Base Processor)
- Next: Task 04 (Audio Transcription Event)
