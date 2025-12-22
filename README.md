# AI Assistant v4 - Multi-Threaded Perception System

A production-ready, event-driven AI assistant framework with multi-threaded perception capabilities. Built with clean architecture principles and designed for extensibility.

## 🎯 Features

- **🔌 Multiple Input Sources**: Camera, text, audio, and extensible to custom sources
- **⚡ Event-Driven Architecture**: Priority-based event queue with pub-sub pattern
- **🧵 Multi-Threaded**: Dedicated threads for each input source and event processing
- **📊 Real-Time Processing**: Non-blocking concurrent processing of multiple inputs
- **🎛️ Dynamic Configuration**: Add/remove input sources at runtime
- **🔒 Thread-Safe**: Built with proper synchronization and lifecycle management
- **📝 Comprehensive Logging**: Color-coded, thread-aware logging system
- **🏗️ Clean Architecture**: Protocol-based interfaces, dependency injection ready
- **🔄 Graceful Shutdown**: Proper resource cleanup and thread termination
- **☁️ Kafka-Ready**: Event bus designed for easy migration to distributed systems

## 🏛️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Perception System                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Input     │  │  Event Bus   │  │   Thread        │   │
│  │   Sources   │→ │  (Priority   │→ │   Manager       │   │
│  │             │  │   Queue)     │  │                 │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  Subscribers  │
                    │  (Your Code)  │
                    └───────────────┘
```

### Directory Structure

```
ai-assistant-v4/
├── src/ai_assistant/
│   ├── perception/           # Perception module
│   │   ├── core/            # Core system & config
│   │   │   ├── config.py   # PerceptionConfig
│   │   │   └── system.py   # PerceptionSystem
│   │   └── input_sources/   # Input source implementations
│   │       ├── base.py     # BaseInputSource (abstract)
│   │       ├── camera.py   # CameraInputSource
│   │       ├── text.py     # TextInputSource
│   │       └── audio.py    # AudioInputSource
│   └── shared/              # Shared utilities
│       ├── interfaces/      # Protocol definitions
│       │   ├── lifecycle.py      # ILifecycle
│       │   ├── event.py          # IEvent, EventPriority
│       │   ├── pubsub.py         # IPublisher, ISubscriber
│       │   ├── event_bus.py      # IEventBus
│       │   └── input_source.py   # IInputSource
│       ├── events/          # Event system
│       │   ├── event.py          # Event dataclass
│       │   ├── specialized.py    # Specialized events
│       │   ├── priority_queue.py # EventPriorityQueue
│       │   └── event_bus.py      # EventBus
│       ├── logging/         # Logging infrastructure
│       │   ├── levels.py         # LogLevel enum
│       │   ├── formatters.py     # Thread-aware formatters
│       │   ├── config.py         # LogConfig, setup
│       │   └── context.py        # Context manager
│       └── threading/       # Thread management
│           ├── thread_info.py    # ThreadInfo
│           ├── thread_pool.py    # ManagedThreadPool
│           └── manager.py        # ThreadManager
├── examples/                # Working examples
│   ├── basic_perception.py      # Single camera demo
│   ├── multi_input.py          # Multiple inputs demo
│   └── dynamic_sources.py      # Dynamic management demo
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
└── tasks/                   # Implementation tasks
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-assistant-v4

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Basic Usage

```python
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.interfaces import IEvent

# Create configuration
config = PerceptionConfig.with_camera(device_id=0, fps=30)

# Create system
system = PerceptionSystem(config)

# Subscribe to events
def handle_frame(event: IEvent):
    frame_data = event.data
    print(f"Received frame: {frame_data['shape']}")

system.subscribe('camera.frame', handle_frame)

# Start the system
system.start()

# Add more sources dynamically
system.add_text_input('console')

# Run for a while...
import time
time.sleep(10)

# Stop gracefully
system.stop()
```

## 📚 Examples

### 1. Basic Perception (Single Camera)

```bash
python examples/basic_perception.py
```

Demonstrates:
- Simple system setup with camera input
- Event subscription and handling
- Graceful shutdown

### 2. Multi-Input Processing

```bash
python examples/multi_input.py
```

Demonstrates:
- Multiple concurrent input sources (camera, text, audio)
- Event statistics tracking
- Priority-based event processing

### 3. Dynamic Source Management

```bash
python examples/dynamic_sources.py
```

Demonstrates:
- Adding/removing sources at runtime
- Interactive command interface
- Source monitoring and management

## 🔧 Configuration

### PerceptionConfig Options

```python
from ai_assistant.perception import PerceptionConfig
from ai_assistant.shared.logging import LogLevel
from pathlib import Path

config = PerceptionConfig(
    # Logging
    log_level=LogLevel.INFO,
    log_file=Path("logs/perception.log"),
    colored_console=True,
    
    # Event bus
    max_queue_size=1000,
    
    # Threading
    max_worker_threads=4,
    
    # Pre-configured input sources
    input_sources=[
        {
            'type': 'camera',
            'source_id': 'camera_0',
            'config': {'device_id': 0, 'fps': 30}
        }
    ]
)
```

### Quick Config Methods

```python
# Default configuration
config = PerceptionConfig.default()

# With camera
config = PerceptionConfig.with_camera(device_id=0, fps=30)
```

## 🎨 Creating Custom Input Sources

Extend `BaseInputSource` to create custom input sources:

```python
from ai_assistant.perception.input_sources import BaseInputSource
from ai_assistant.shared.events import Event
from ai_assistant.shared.interfaces import EventPriority

class CustomInputSource(BaseInputSource):
    """Custom input source implementation."""
    
    def __init__(self, source_id: str, event_bus, config=None):
        super().__init__(
            source_id=source_id,
            source_type="custom",
            event_bus=event_bus,
            config=config
        )
    
    def _initialize_source(self) -> None:
        """Initialize your custom source."""
        # Setup code here
        pass
    
    def _cleanup_source(self) -> None:
        """Clean up your custom source."""
        # Cleanup code here
        pass
    
    def _capture_and_publish(self) -> None:
        """Capture data and publish events."""
        # Get data from your source
        data = self._get_custom_data()
        
        # Create and publish event
        event = Event.create(
            event_type="custom.data",
            source=self.source_id,
            data=data,
            priority=EventPriority.NORMAL
        )
        self._publish_event(event)
    
    def _get_custom_data(self):
        """Your custom data acquisition logic."""
        return {"value": 42}

# Register with system
system.add_input_source(CustomInputSource('my_source', system._event_bus))
```

## 📊 Event Types

### Built-in Event Types

| Event Type | Priority | Source | Data Fields |
|------------|----------|--------|-------------|
| `camera.frame` | NORMAL | CameraInputSource | `frame`, `frame_number`, `shape`, `timestamp` |
| `text.input` | HIGH | TextInputSource | `text`, `timestamp` |
| `audio.sample` | NORMAL | AudioInputSource | `samples`, `duration`, `sample_rate` |

### Event Priority Levels

```python
from ai_assistant.shared.interfaces import EventPriority

EventPriority.LOW       # Background tasks
EventPriority.NORMAL    # Default priority
EventPriority.HIGH      # User input, important events
EventPriority.CRITICAL  # System-critical events
```

Events are processed in priority order: CRITICAL → HIGH → NORMAL → LOW

## 🧵 Thread Management

The system manages multiple threads automatically:

- **EventBus Thread**: Processes events from the priority queue
- **Input Source Threads**: One thread per input source for data capture
- **Thread Pool**: Configurable worker threads for async operations

### Thread Safety

All components are thread-safe and use proper synchronization:
- `threading.RLock` for recursive locking
- `queue.PriorityQueue` for thread-safe event queuing
- Atomic operations for state management

## 📝 Logging

### Log Levels

```python
from ai_assistant.shared.logging import LogLevel, setup_logging, LogConfig

config = LogConfig(
    level=LogLevel.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file=Path("app.log"),
    console_output=True,
    colored_console=True
)
setup_logging(config)
```

### Thread-Aware Logging

All log messages include thread information:

```
2024-01-15 10:30:45 [MainThread-123456] INFO: System starting
2024-01-15 10:30:46 [EventBus-789012] DEBUG: Processing event
2024-01-15 10:30:47 [camera_0-345678] INFO: Frame captured
```

### Context-Based Logging

```python
from ai_assistant.shared.logging import log_context, get_logger

logger = get_logger(__name__)

with log_context(user_id="user123", request_id="req456"):
    logger.info("Processing request")  # Includes context in log
```

## 🔄 Lifecycle Management

All components follow the `ILifecycle` protocol:

```python
# Initialize (setup resources)
system.initialize()

# Start (begin processing)
system.start()

# Check status
if system.is_running():
    print("System is running")

# Stop (graceful shutdown)
system.stop()
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_assistant --cov-report=html

# Run specific test file
pytest tests/unit/test_event_bus.py

# Type checking
mypy src/

# Linting
ruff check src/
```

## 📦 Dependencies

### Core Dependencies
- Python 3.11+
- opencv-python (camera input)
- numpy (data processing)
- pillow (image handling)

### Development Dependencies
- pytest (testing)
- mypy (type checking)
- ruff (linting)
- pytest-cov (coverage)

## 🎯 Design Principles

1. **Interface-Driven Design**: All components depend on Protocol interfaces
2. **Separation of Concerns**: Clear boundaries between modules
3. **Single Responsibility**: Each class has one well-defined purpose
4. **Open/Closed**: Open for extension, closed for modification
5. **Dependency Inversion**: Depend on abstractions, not concretions

## 🛣️ Roadmap

### Completed ✅
- [x] Core interfaces and protocols
- [x] Event system with priority queue
- [x] Thread management
- [x] Logging infrastructure
- [x] Input source framework
- [x] Perception system integration
- [x] Example applications

### In Progress 🚧
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Documentation site

### Planned 📋
- [ ] Kafka integration for distributed events
- [ ] Additional input sources (network, files, sensors)
- [ ] Event replay and debugging tools
- [ ] Metrics and monitoring dashboard
- [ ] Plugin system for extensions
- [ ] Docker containerization

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the existing code style and architecture
2. Add tests for new features
3. Update documentation
4. Ensure all tests and type checks pass

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with modern Python practices and inspired by:
- Reactive programming principles
- Clean Architecture (Robert C. Martin)
- Domain-Driven Design
- Enterprise Integration Patterns

---

**Status**: Production Ready
**Version**: 4.0.0
**Python**: 3.11+
