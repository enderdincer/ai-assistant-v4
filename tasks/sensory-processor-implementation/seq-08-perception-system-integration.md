# Task 08: Integrate Processors with PerceptionSystem

## High-Level Objective
Extend the PerceptionSystem to support optional processor management, allowing processors to be registered, configured, and managed alongside input sources while maintaining backward compatibility.

## Core Direction
Processors should be first-class citizens in the perception system but remain optional. The system should work without processors (existing functionality preserved) and seamlessly incorporate processors when configured.

## Dependencies
- Task 02: BaseProcessor must be complete
- Task 03: ProcessorManager must be complete
- Task 06: STTProcessor should be implemented (for testing)

## Prerequisites
- PerceptionSystem implementation understood
- PerceptionConfig pattern understood
- ProcessorManager available

## Acceptance Criteria

### 1. System Integration
- [ ] Add ProcessorManager to PerceptionSystem
- [ ] Initialize ProcessorManager in `__init__()`
- [ ] Integrate processor lifecycle with system lifecycle
- [ ] Register ProcessorManager with ThreadManager for monitoring

### 2. Configuration Support
- [ ] Extend PerceptionConfig to support processor definitions
- [ ] Add `processors: List[Dict[str, Any]]` field to config
- [ ] Support processor-specific configuration
- [ ] Create helper methods: `with_stt_processor()`, etc.

### 3. System Lifecycle Methods
- [ ] Start processors when system starts (after event bus)
- [ ] Stop processors when system stops (before event bus)
- [ ] Ensure proper initialization order:
  1. Event bus
  2. Thread manager
  3. Processors (subscribe to events)
  4. Input sources (publish events)

### 4. Processor Management API
- [ ] `add_processor(processor: IProcessor) -> None` - Add processor dynamically
- [ ] `remove_processor(processor_id: str) -> None` - Remove processor
- [ ] `get_processor(processor_id: str) -> Optional[IProcessor]` - Get processor by ID
- [ ] `list_processors() -> Dict[str, str]` - List active processors

### 5. Status Reporting
- [ ] Include processor status in `get_status()` method
- [ ] Show processor count and states
- [ ] Include in thread manager status

### 6. Backward Compatibility
- [ ] System works without any processors configured
- [ ] Existing examples continue to work unchanged
- [ ] No breaking changes to public API
- [ ] Default config has no processors (empty list)

## Implementation Notes

### File Modifications
- Modify: `src/ai_assistant/perception/core/system.py`
- Modify: `src/ai_assistant/perception/core/config.py`
- Modify: `src/ai_assistant/perception/__init__.py` (exports)

### PerceptionSystem Changes

#### Add ProcessorManager
```python
from ai_assistant.perception.processors import ProcessorManager

class PerceptionSystem:
    def __init__(self, config: Optional[PerceptionConfig] = None) -> None:
        # ... existing initialization ...
        
        # Create processor manager
        self._processor_manager = ProcessorManager()
        
        # Register with thread manager
        self._thread_manager.register_component("processor_manager", self._processor_manager)
```

#### Update Lifecycle
```python
def start(self) -> None:
    """Start the perception system."""
    logger.info("Starting perception system")
    
    # Initialize and start core components
    self._thread_manager.initialize_all()
    self._thread_manager.start_all()
    
    # Create and start processors from configuration
    for proc_config in self._config.processors:
        proc_type = proc_config["type"]
        proc_id = proc_config["processor_id"]
        config = proc_config.get("config", {})
        
        if proc_type == "stt":
            self.add_stt_processor(proc_id, config)
        else:
            logger.warning(f"Unknown processor type: {proc_type}")
    
    # Create and start input sources
    for source_config in self._config.input_sources:
        # ... existing input source creation ...
    
    logger.info("Perception system started")

def stop(self) -> None:
    """Stop the perception system."""
    logger.info("Stopping perception system")
    
    # Stop input sources first (stop publishing events)
    for source_id, source in list(self._input_sources.items()):
        self.remove_input_source(source_id)
    
    # Stop processors (stop consuming events)
    self._processor_manager.stop_all()
    
    # Stop core components
    self._thread_manager.stop_all()
    
    logger.info("Perception system stopped")
```

#### Add Processor Methods
```python
def add_stt_processor(
    self, processor_id: str, config: Optional[Dict[str, Any]] = None
) -> STTProcessor:
    """Add an STT processor dynamically."""
    from ai_assistant.perception.processors import STTProcessor
    
    logger.info(f"Adding STT processor: {processor_id}")
    
    processor = STTProcessor(processor_id, self._event_bus, config)
    processor.initialize()
    processor.start()
    
    self._processor_manager.register_processor(processor)
    
    return processor

def add_processor(self, processor: IProcessor) -> None:
    """Add a processor dynamically."""
    logger.info(f"Adding processor: {processor.processor_id}")
    
    processor.initialize()
    processor.start()
    
    self._processor_manager.register_processor(processor)

def remove_processor(self, processor_id: str) -> None:
    """Remove a processor."""
    processor = self._processor_manager.get_processor(processor_id)
    if processor:
        processor.stop()
        self._processor_manager.unregister_processor(processor_id)

def get_processor(self, processor_id: str) -> Optional[IProcessor]:
    """Get a processor by ID."""
    return self._processor_manager.get_processor(processor_id)

def list_processors(self) -> Dict[str, str]:
    """List all active processors."""
    return self._processor_manager.list_processors()
```

#### Update Status
```python
def get_status(self) -> Dict[str, Any]:
    """Get the status of the perception system."""
    return {
        "running": self._thread_manager.is_running(),
        "event_bus": {
            "running": self._event_bus.is_running(),
            "queue_size": self._event_bus.get_queue_size(),
        },
        "processors": self._processor_manager.get_status(),
        "input_sources": {
            source_id: {
                "type": source.source_type,
                "running": source.is_running(),
            }
            for source_id, source in self._input_sources.items()
        },
        "thread_manager": self._thread_manager.get_status(),
    }
```

### PerceptionConfig Changes

```python
@dataclass
class PerceptionConfig:
    """Configuration for the perception system."""
    
    # ... existing fields ...
    
    # Processor configurations
    processors: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def with_stt_processor(
        cls,
        processor_id: str = "stt_default",
        model_name: str = "nvidia/canary-1b",
        device: str = "cpu",
        **kwargs: Any,
    ) -> "PerceptionConfig":
        """Create configuration with an STT processor.
        
        Args:
            processor_id: Unique identifier for the processor
            model_name: HuggingFace model name
            device: Inference device (cpu/cuda/mps)
            **kwargs: Additional processor configuration
        
        Returns:
            PerceptionConfig: Configuration with STT processor
        """
        config = cls.default()
        config.processors.append(
            {
                "type": "stt",
                "processor_id": processor_id,
                "config": {
                    "model_name": model_name,
                    "device": device,
                    **kwargs,
                },
            }
        )
        return config
    
    @classmethod
    def with_audio_and_stt(
        cls,
        audio_source_id: str = "microphone",
        stt_processor_id: str = "stt",
        device: str = "cpu",
        **kwargs: Any,
    ) -> "PerceptionConfig":
        """Create configuration with audio input and STT processor.
        
        This is a convenience method for the common use case of
        capturing audio and transcribing it.
        
        Args:
            audio_source_id: Audio input source identifier
            stt_processor_id: STT processor identifier
            device: Inference device
            **kwargs: Additional configuration
        
        Returns:
            PerceptionConfig: Complete audio + STT configuration
        """
        config = cls.default()
        
        # Add audio input
        config.input_sources.append(
            {
                "type": "audio",
                "source_id": audio_source_id,
                "config": kwargs.get("audio_config", {}),
            }
        )
        
        # Add STT processor
        config.processors.append(
            {
                "type": "stt",
                "processor_id": stt_processor_id,
                "config": {
                    "model_name": kwargs.get("model_name", "nvidia/canary-1b"),
                    "device": device,
                    "buffer_duration": kwargs.get("buffer_duration", 3.0),
                },
            }
        )
        
        return config
```

## Validation

### Unit Tests
- [ ] Test system initialization with processors
- [ ] Test system start/stop with processors
- [ ] Test adding/removing processors dynamically
- [ ] Test backward compatibility (no processors)
- [ ] Test status reporting includes processors

### Integration Tests
- [ ] Test full system with audio input and STT processor
- [ ] Verify events flow from input → processor → subscribers
- [ ] Test system shutdown cleans up processors
- [ ] Test error handling (processor failures don't crash system)

### Test File Locations
- `tests/unit/perception/test_perception_system_processors.py`
- `tests/integration/perception/test_stt_integration.py`

## Backward Compatibility Verification

Test that existing examples still work:
- [ ] `examples/basic_perception.py` - No changes required
- [ ] `examples/dynamic_sources.py` - No changes required
- [ ] `examples/multi_input.py` - No changes required

## Success Metrics
- Processors integrate seamlessly with PerceptionSystem
- Backward compatibility maintained (all existing tests pass)
- New processor tests pass
- Configuration is intuitive and well-documented
- System lifecycle handles processors correctly

## Estimated Effort
Medium (3-4 hours)

## Related Tasks
- Previous: Task 03 (Processor Manager), Task 06 (STT Processor)
- Next: Task 09 (Example Implementation)
