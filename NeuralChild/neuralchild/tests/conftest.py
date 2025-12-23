"""Common test fixtures for the NeuralChild project.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
import torch
import tempfile
import shutil
import os
from typing import Dict, Any, Generator
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

from neuralchild.config import Config
from neuralchild.mind.mind_core import Mind
from neuralchild.mother.mother_llm import MotherLLM
from neuralchild.core.schemas import (
    Memory, Belief, Need, DevelopmentalStage,
    NetworkMessage, VectorOutput, TextOutput
)
from neuralchild.communication.message_bus import MessageBus, MessageFilter


@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    config = Config()
    # Set simulate_llm to True to avoid making real API calls during tests
    config.development.simulate_llm = True
    config.development.debug_mode = True
    config.mind.development_acceleration = 10.0  # Speed up tests
    config.logging.level = "ERROR"  # Reduce noise during tests
    return config


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config_file(temp_dir: str, config: Config) -> str:
    """Create a temporary config file."""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    config.to_yaml(config_path)
    return config_path


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _mock_response(content: str = None):
        if content is None:
            content = '{"understanding": "Child is calm", "response": "Hello, my child", "action": "speak", "development_focus": "language"}'
        return {"choices": [{"message": {"content": content}}]}
    return _mock_response


@pytest.fixture
def mind(config: Config) -> Mind:
    """Create a test Mind instance."""
    # Patch config to use test settings
    with patch('neuralchild.mind.mind_core.config', config):
        mind = Mind()
        # Initialize essential attributes
        mind.state.developmental_stage = DevelopmentalStage.INFANT
        mind.state.energy_level = 0.8
        return mind


@pytest.fixture
def mother(config: Config) -> MotherLLM:
    """Create a test MotherLLM instance."""
    return MotherLLM()


@pytest.fixture
def message_bus() -> Generator[MessageBus, None, None]:
    """Create a test MessageBus instance."""
    bus = MessageBus()
    yield bus
    # Cleanup
    bus.running = False
    if hasattr(bus, 'delivery_thread') and bus.delivery_thread.is_alive():
        bus.delivery_thread.join(timeout=1)


@pytest.fixture
def sample_memory() -> Memory:
    """Create a sample memory for testing."""
    return Memory(
        id="test_memory_1",
        content={"type": "interaction", "data": "saw a ball"},
        creation_time=datetime.now(),
        strength=1.0,
        emotional_valence=0.5,
        tags=["visual", "object"],
        developmental_stage=DevelopmentalStage.INFANT
    )


@pytest.fixture
def sample_belief() -> Belief:
    """Create a sample belief for testing."""
    return Belief(
        subject="ball",
        predicate="is",
        object="round",
        confidence=0.7,
        supporting_memories=["test_memory_1"],
        developmental_stage=DevelopmentalStage.INFANT
    )


@pytest.fixture
def sample_need() -> Need:
    """Create a sample need for testing."""
    return Need(
        name="attention",
        intensity=0.6,
        satisfaction_level=0.4
    )


@pytest.fixture
def sample_network_message() -> NetworkMessage:
    """Create a sample network message for testing."""
    return NetworkMessage(
        sender="perception",
        receiver="consciousness",
        content={"data": [0.1, 0.2, 0.3]},
        message_type="sensory",
        priority=0.8,
        developmental_stage=DevelopmentalStage.INFANT
    )


@pytest.fixture
def test_input() -> Dict[str, Any]:
    """Create a test input for the Mind."""
    return {
        "visual": [0.1] * 64,
        "auditory": [0.2] * 64,
        "language": "Hello, child",
        "source": "test"
    }


@pytest.fixture
def test_tensor() -> torch.Tensor:
    """Create a test tensor for network input."""
    return torch.randn(1, 64)


@pytest.fixture
def mock_chat_completion(mock_llm_response):
    """Mock the chat_completion function for testing."""
    with patch('neuralchild.utils.llm_module.chat_completion', return_value=mock_llm_response()) as mock:
        yield mock


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture
def sample_vector_output() -> VectorOutput:
    """Create a sample vector output for testing."""
    return VectorOutput(
        source="perception",
        data=[0.1, 0.2, 0.3, 0.4, 0.5],
        developmental_stage=DevelopmentalStage.INFANT
    )


@pytest.fixture
def sample_text_output() -> TextOutput:
    """Create a sample text output for testing."""
    return TextOutput(
        source="consciousness",
        text="I see something",
        confidence=0.8,
        developmental_stage=DevelopmentalStage.INFANT
    )


@pytest.fixture
def message_filter() -> MessageFilter:
    """Create a sample message filter for testing."""
    return MessageFilter(
        sender="perception",
        message_type="sensory"
    )


# Helper functions for tests
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert that a tensor has the expected shape."""
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_valid_probability(value: float):
    """Assert that a value is a valid probability (0-1)."""
    assert 0.0 <= value <= 1.0, f"Expected probability between 0 and 1, got {value}"
