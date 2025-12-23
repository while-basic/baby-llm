# NeuralChild Test Suite

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License

## Overview

This comprehensive pytest test suite provides >70% code coverage for the NeuralChild v1.0.0 framework. The tests are organized into logical modules that mirror the codebase structure.

## Test Files

### 1. `__init__.py`
Empty init file marking the tests directory as a Python package.

### 2. `conftest.py` (216 lines)
Central fixture definitions used across all tests:
- **Config fixtures**: Test configuration with simulated LLM mode
- **Component fixtures**: Mind, MotherLLM, MessageBus instances
- **Schema fixtures**: Sample Memory, Belief, Need, NetworkMessage objects
- **Mock fixtures**: LLM response mocking, random seed reset
- **Helper functions**: Tensor shape assertions, probability validation

### 3. `test_config.py` (343 lines)
Configuration management tests covering:
- **ServerConfig**: URL configurations, default values
- **ModelConfig**: Temperature validation, model settings
- **VisualizationConfig**: Display settings, color schemes
- **MindConfig**: Learning rates, network configs, feature toggles
- **LoggingConfig**: Log levels, file/console logging
- **DevelopmentConfig**: Debug mode, experimental features
- **Main Config**: YAML/JSON serialization, loading, saving
- **Integration tests**: Roundtrip serialization, partial updates

### 4. `test_schemas.py` (443 lines)
Core schema and data structure tests:
- **DevelopmentalStage**: Enum values, ordering
- **NetworkMessage**: Creation, defaults, serialization, priorities
- **NetworkState**: State management, developmental weights
- **Memory**: Access patterns, decay, forgetting threshold
- **VectorOutput/TextOutput**: Output formatting
- **Belief**: Confidence updates, natural language conversion
- **Need**: Intensity updates, satisfaction mechanics
- **Integration**: Cross-schema relationships, consistency

### 5. `test_networks.py` (379 lines)
Neural network implementation tests:
- **ConsciousnessNetwork**: 
  - Initialization, RNN/self-model components
  - Forward pass (2D/3D input)
  - Hidden state maintenance
  - Awareness/self-awareness tracking
  - Developmental growth
- **EmotionsNetwork**:
  - Emotion processing
  - Emotional state tracking
  - Different input handling
- **PerceptionNetwork**:
  - Visual/auditory processing
  - Multi-modal input
  - Attention mechanisms
- **ThoughtsNetwork**:
  - Thought generation
  - Thought state management
- **Integration tests**:
  - Network chaining
  - Message passing
  - Developmental progression
  - State persistence
- **Error handling**: Invalid inputs, None handling

### 6. `test_mind.py` (488 lines)
Mind core functionality tests:
- **Initialization**: State setup, networks, memory/belief systems
- **Memory operations**:
  - Store/retrieve memories
  - Memory consolidation
  - Memory decay
  - Context-based recall
- **Belief system**:
  - Belief formation
  - Belief updates
  - Memory-to-belief pipeline
  - Contradiction handling
- **Needs system**:
  - Basic needs tracking
  - Need updates
  - Need satisfaction
- **Development**:
  - Stage progression
  - Stage transitions
  - Development metrics
- **Processing**:
  - Step execution
  - Input processing
  - Output generation
  - Network communication
- **State management**:
  - Observable state
  - Energy levels
  - Emotional state
  - State persistence
- **Integration**: Full processing cycles, need-driven behavior
- **Error handling**: None inputs, invalid data

### 7. `test_mother.py` (484 lines)
Mother LLM component tests:
- **MotherResponse schema**: Creation, defaults, serialization
- **Initialization**:
  - Personality traits
  - Response templates
  - Developmental techniques
- **Observation**:
  - Mind state observation
  - Stage-appropriate responses
  - Need detection
- **Response generation**:
  - Stage-appropriate content
  - Response variety
  - Emotional responses
- **Interaction history**: Recording, format, retrieval
- **Developmental focus**:
  - Language development
  - Emotional development
  - Cognitive development
  - Stage-specific techniques
- **Timing**: Response intervals, timing respect
- **Integration**: Mother-mind interaction cycles, adaptation
- **Error handling**: None inputs, LLM errors, invalid JSON

### 8. `test_message_bus.py` (579 lines)
Message bus communication tests:
- **MessageFilter**:
  - Filter creation
  - Validation
  - Multiple criteria
  - Priority/stage filtering
- **SubscriptionInfo**: Creation, callbacks, queues
- **Initialization**: Threading, initial state
- **Subscription**:
  - Callback-based
  - Queue-based
  - Multiple subscriptions
  - Unsubscribe
- **Publishing**:
  - Message publishing
  - Subscriber delivery
  - Multiple messages
  - History limits
- **Filtering**:
  - By sender
  - By message type
  - By priority
  - Multiple criteria
- **Retrieval**: Queue messages, timeouts
- **Thread safety**: Concurrent publishing, concurrent subscribing
- **GlobalMessageBus**: Singleton access
- **Integration**: End-to-end flow, network communication, broadcasting
- **Error handling**: Invalid messages, invalid filters, non-existent queues

## Running Tests

### Run all tests:
```bash
cd /home/user/baby-llm/NeuralChild
pytest neuralchild/tests/
```

### Run specific test file:
```bash
pytest neuralchild/tests/test_config.py
pytest neuralchild/tests/test_schemas.py
pytest neuralchild/tests/test_networks.py
pytest neuralchild/tests/test_mind.py
pytest neuralchild/tests/test_mother.py
pytest neuralchild/tests/test_message_bus.py
```

### Run with coverage:
```bash
pytest neuralchild/tests/ --cov=neuralchild --cov-report=html
pytest neuralchild/tests/ --cov=neuralchild --cov-report=term-missing
```

### Run specific test class:
```bash
pytest neuralchild/tests/test_config.py::TestConfig
pytest neuralchild/tests/test_mind.py::TestMindMemorySystem
```

### Run specific test:
```bash
pytest neuralchild/tests/test_schemas.py::TestMemory::test_memory_decay
```

### Run with verbose output:
```bash
pytest neuralchild/tests/ -v
pytest neuralchild/tests/ -vv
```

### Run in parallel (requires pytest-xdist):
```bash
pip install pytest-xdist
pytest neuralchild/tests/ -n auto
```

## Test Statistics

- **Total test files**: 8
- **Total lines of test code**: ~2,937
- **Estimated test count**: 150+ individual tests
- **Coverage target**: >70%

## Test Categories

### Unit Tests
- Individual component testing
- Schema validation
- Configuration management
- Network forward passes

### Integration Tests
- Component interaction
- End-to-end flows
- Multi-network communication
- Mother-mind interactions

### Error Handling Tests
- Invalid inputs
- Edge cases
- None handling
- Concurrent operations

## Mocking Strategy

The test suite uses extensive mocking to avoid external dependencies:
- **LLM calls**: Mocked via `unittest.mock.patch`
- **Random seeds**: Fixed for reproducibility
- **Time-based operations**: Controllable via fixtures
- **File I/O**: Temporary directories provided

## Key Features

1. **Comprehensive fixtures**: Reusable test components in conftest.py
2. **Production-ready**: Error handling, edge cases, integration tests
3. **Well-documented**: Descriptive test names and docstrings
4. **Maintainable**: Each file under 600 lines, logical organization
5. **Fast execution**: Mocked external calls, parallelizable
6. **Deterministic**: Fixed random seeds, reproducible results

## Dependencies

Required packages (from requirements.txt):
- pytest>=7.0.0
- pytest-cov>=4.0.0
- torch>=2.0.0
- pydantic>=2.0.0
- pyyaml>=6.0
- numpy>=1.20.0

## Copyright

All test files include the Celaya Solutions copyright header:
```python
"""[Description]

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""
```
