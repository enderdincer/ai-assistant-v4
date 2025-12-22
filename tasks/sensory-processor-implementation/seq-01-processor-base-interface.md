# Task 01: Create Base Processor Interface

## High-Level Objective
Define the foundational interface and abstract base class for sensory processors that can subscribe to raw input events and publish processed events.

## Core Direction
Processors are optional plugins that operate independently of input sources. They should follow the same lifecycle patterns as existing components (initialize, start, stop) and integrate seamlessly with the event bus architecture.

## Dependencies
- None (foundational task)

## Prerequisites
- Existing event bus architecture (already implemented)
- Existing lifecycle interfaces (already implemented)

## Acceptance Criteria

### 1. Interface Definition
- [ ] Create `IProcessor` interface in `src/ai_assistant/shared/interfaces/processor.py`
- [ ] Interface must include:
  - `processor_id: str` property
  - `processor_type: str` property
  - `input_event_types: List[str]` property (event types to subscribe to)
  - `output_event_types: List[str]` property (event types produced)
  - `process(event: IEvent) -> List[IEvent]` method (sync processing)
  - `process_async(event: IEvent) -> Awaitable[List[IEvent]]` method (async processing)
- [ ] Interface must extend `ILifecycle` for standard lifecycle management

### 2. Export Interface
- [ ] Add `IProcessor` to `src/ai_assistant/shared/interfaces/__init__.py`
- [ ] Ensure proper type hints and documentation

### 3. Validation
- [ ] Interface compiles without errors
- [ ] mypy type checking passes
- [ ] Documentation is clear and follows project conventions

## Implementation Notes

### File Structure
```
src/ai_assistant/shared/interfaces/
├── __init__.py          # Export IProcessor
├── processor.py         # NEW: IProcessor interface
└── (existing files...)
```

### Key Design Decisions
- Processors return `List[IEvent]` to allow:
  - Returning empty list (filtering)
  - Returning single event (1:1 transformation)
  - Returning multiple events (1:N expansion)
- Both sync and async processing methods for flexibility
- Processors declare their input/output event types for transparency

### Code Pattern
```python
from typing import Protocol, List, Awaitable
from ai_assistant.shared.interfaces import ILifecycle, IEvent

class IProcessor(ILifecycle, Protocol):
    """Interface for sensory processors."""
    
    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        ...
    
    @property
    def processor_type(self) -> str:
        """Type/category of processor (e.g., 'stt', 'object_detection')."""
        ...
    
    @property
    def input_event_types(self) -> List[str]:
        """Event types this processor subscribes to."""
        ...
    
    @property
    def output_event_types(self) -> List[str]:
        """Event types this processor produces."""
        ...
    
    def process(self, event: IEvent) -> List[IEvent]:
        """Process an event synchronously."""
        ...
    
    async def process_async(self, event: IEvent) -> List[IEvent]:
        """Process an event asynchronously."""
        ...
```

## Success Metrics
- Interface defined with all required methods
- Type checking passes
- Ready for concrete implementation in next task

## Estimated Effort
Small (< 1 hour)

## Related Tasks
- Next: Task 02 (Base Processor Implementation)
