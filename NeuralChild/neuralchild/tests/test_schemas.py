"""Tests for core schemas and data structures.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from neuralchild.core.schemas import (
    DevelopmentalStage,
    NetworkMessage,
    NetworkState,
    Memory,
    VectorOutput,
    TextOutput,
    Belief,
    Need
)


class TestDevelopmentalStage:
    """Test suite for DevelopmentalStage enum."""

    def test_developmental_stages_exist(self):
        """Test that all expected developmental stages exist."""
        assert DevelopmentalStage.INFANT.value == 1
        assert DevelopmentalStage.TODDLER.value == 2
        assert DevelopmentalStage.CHILD.value == 3
        assert DevelopmentalStage.ADOLESCENT.value == 4
        assert DevelopmentalStage.MATURE.value == 5

    def test_developmental_stage_ordering(self):
        """Test that developmental stages are properly ordered."""
        assert DevelopmentalStage.INFANT.value < DevelopmentalStage.TODDLER.value
        assert DevelopmentalStage.TODDLER.value < DevelopmentalStage.CHILD.value
        assert DevelopmentalStage.CHILD.value < DevelopmentalStage.ADOLESCENT.value
        assert DevelopmentalStage.ADOLESCENT.value < DevelopmentalStage.MATURE.value


class TestNetworkMessage:
    """Test suite for NetworkMessage schema."""

    def test_network_message_creation(self, sample_network_message: NetworkMessage):
        """Test creating a network message."""
        assert sample_network_message.sender == "perception"
        assert sample_network_message.receiver == "consciousness"
        assert sample_network_message.message_type == "sensory"
        assert sample_network_message.priority == 0.8

    def test_network_message_defaults(self):
        """Test network message default values."""
        msg = NetworkMessage(
            sender="test",
            receiver="target",
            content={"data": "value"}
        )
        assert msg.message_type == "standard"
        assert msg.priority == 1.0
        assert msg.developmental_stage == DevelopmentalStage.INFANT
        assert isinstance(msg.timestamp, datetime)

    def test_network_message_to_dict(self, sample_network_message: NetworkMessage):
        """Test converting network message to dictionary."""
        msg_dict = sample_network_message.to_dict()
        assert isinstance(msg_dict, dict)
        assert msg_dict["sender"] == "perception"
        assert msg_dict["receiver"] == "consciousness"
        assert "timestamp" in msg_dict
        assert "content" in msg_dict

    def test_network_message_priority_levels(self):
        """Test different priority levels."""
        low_priority = NetworkMessage(
            sender="test", receiver="target",
            content={}, priority=0.1
        )
        high_priority = NetworkMessage(
            sender="test", receiver="target",
            content={}, priority=1.0
        )
        assert low_priority.priority < high_priority.priority


class TestNetworkState:
    """Test suite for NetworkState schema."""

    def test_network_state_creation(self):
        """Test creating a network state."""
        state = NetworkState(
            name="test_network",
            active=True,
            parameters={"param1": 0.5}
        )
        assert state.name == "test_network"
        assert state.active is True
        assert state.parameters["param1"] == 0.5

    def test_network_state_defaults(self):
        """Test network state default values."""
        state = NetworkState(name="test")
        assert state.active is True
        assert isinstance(state.last_update, datetime)
        assert isinstance(state.parameters, dict)
        assert isinstance(state.developmental_weights, dict)

    def test_network_state_to_dict(self):
        """Test converting network state to dictionary."""
        state = NetworkState(
            name="test",
            parameters={"test": 1.0},
            developmental_weights={DevelopmentalStage.INFANT: 0.5}
        )
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["name"] == "test"
        assert "last_update" in state_dict
        assert "INFANT" in state_dict["developmental_weights"]


class TestMemory:
    """Test suite for Memory schema."""

    def test_memory_creation(self, sample_memory: Memory):
        """Test creating a memory."""
        assert sample_memory.id == "test_memory_1"
        assert sample_memory.strength == 1.0
        assert sample_memory.emotional_valence == 0.5
        assert "visual" in sample_memory.tags

    def test_memory_defaults(self):
        """Test memory default values."""
        memory = Memory(
            id="test",
            content={"data": "test"}
        )
        assert memory.strength == 1.0
        assert memory.emotional_valence == 0.0
        assert isinstance(memory.tags, list)
        assert memory.developmental_stage == DevelopmentalStage.INFANT

    def test_memory_access(self):
        """Test accessing a memory strengthens it."""
        memory = Memory(id="test", content={}, strength=0.5)
        original_strength = memory.strength
        original_time = memory.last_access_time

        # Access the memory
        memory.access()

        # Check that strength increased
        assert memory.strength > original_strength
        assert memory.last_access_time > original_time

    def test_memory_decay(self):
        """Test memory decay over time."""
        memory = Memory(id="test", content={}, strength=1.0)

        # Decay the memory
        memory.decay(amount=0.3)
        assert memory.strength == 0.7

        # Decay cannot go below 0
        memory.decay(amount=1.0)
        assert memory.strength == 0.0

    def test_memory_is_forgotten(self):
        """Test checking if memory is forgotten."""
        memory = Memory(id="test", content={}, strength=1.0)
        assert not memory.is_forgotten()

        # Decay until forgotten
        memory.strength = 0.05
        assert memory.is_forgotten()

    def test_memory_emotional_valence_range(self):
        """Test emotional valence is within expected range."""
        positive_memory = Memory(
            id="test1", content={}, emotional_valence=0.8
        )
        negative_memory = Memory(
            id="test2", content={}, emotional_valence=-0.8
        )
        neutral_memory = Memory(
            id="test3", content={}, emotional_valence=0.0
        )

        assert positive_memory.emotional_valence > 0
        assert negative_memory.emotional_valence < 0
        assert neutral_memory.emotional_valence == 0


class TestVectorOutput:
    """Test suite for VectorOutput schema."""

    def test_vector_output_creation(self, sample_vector_output: VectorOutput):
        """Test creating a vector output."""
        assert sample_vector_output.source == "perception"
        assert isinstance(sample_vector_output.data, list)
        assert len(sample_vector_output.data) == 5

    def test_vector_output_defaults(self):
        """Test vector output default values."""
        output = VectorOutput(
            source="test",
            data=[0.1, 0.2, 0.3]
        )
        assert isinstance(output.timestamp, datetime)
        assert output.developmental_stage == DevelopmentalStage.INFANT


class TestTextOutput:
    """Test suite for TextOutput schema."""

    def test_text_output_creation(self, sample_text_output: TextOutput):
        """Test creating a text output."""
        assert sample_text_output.source == "consciousness"
        assert sample_text_output.text == "I see something"
        assert sample_text_output.confidence == 0.8

    def test_text_output_defaults(self):
        """Test text output default values."""
        output = TextOutput(
            source="test",
            text="test output"
        )
        assert output.confidence == 1.0
        assert isinstance(output.timestamp, datetime)
        assert output.developmental_stage == DevelopmentalStage.INFANT


class TestBelief:
    """Test suite for Belief schema."""

    def test_belief_creation(self, sample_belief: Belief):
        """Test creating a belief."""
        assert sample_belief.subject == "ball"
        assert sample_belief.predicate == "is"
        assert sample_belief.object == "round"
        assert sample_belief.confidence == 0.7

    def test_belief_defaults(self):
        """Test belief default values."""
        belief = Belief(
            subject="test",
            predicate="is",
            object="example"
        )
        assert belief.confidence == 0.5
        assert isinstance(belief.creation_time, datetime)
        assert isinstance(belief.supporting_memories, list)
        assert belief.developmental_stage == DevelopmentalStage.INFANT

    def test_belief_update_confidence(self):
        """Test updating belief confidence."""
        belief = Belief(
            subject="test",
            predicate="is",
            object="example",
            confidence=0.5
        )
        original_time = belief.last_update_time

        # Update with positive evidence
        belief.update_confidence(0.8)
        assert belief.confidence > 0.5
        assert belief.last_update_time > original_time

    def test_belief_to_natural_language(self):
        """Test converting belief to natural language."""
        # High confidence belief
        strong_belief = Belief(
            subject="sky",
            predicate="is",
            object="blue",
            confidence=0.9
        )
        assert "sure" in strong_belief.to_natural_language().lower()

        # Medium confidence belief
        medium_belief = Belief(
            subject="sky",
            predicate="is",
            object="blue",
            confidence=0.6
        )
        assert "think" in medium_belief.to_natural_language().lower()

        # Low confidence belief
        weak_belief = Belief(
            subject="sky",
            predicate="is",
            object="blue",
            confidence=0.3
        )
        assert "not sure" in weak_belief.to_natural_language().lower()

    def test_belief_supporting_memories(self):
        """Test belief with supporting memories."""
        belief = Belief(
            subject="test",
            predicate="is",
            object="example",
            supporting_memories=["mem1", "mem2", "mem3"]
        )
        assert len(belief.supporting_memories) == 3
        assert "mem1" in belief.supporting_memories


class TestNeed:
    """Test suite for Need schema."""

    def test_need_creation(self, sample_need: Need):
        """Test creating a need."""
        assert sample_need.name == "attention"
        assert sample_need.intensity == 0.6
        assert sample_need.satisfaction_level == 0.4

    def test_need_defaults(self):
        """Test need default values."""
        need = Need(name="test")
        assert need.intensity == 0.5
        assert need.satisfaction_level == 0.5
        assert isinstance(need.last_update, datetime)

    def test_need_validation(self):
        """Test need field validation."""
        # Valid need
        need = Need(name="test", intensity=0.8, satisfaction_level=0.2)
        assert need.intensity == 0.8

        # Intensity out of range
        with pytest.raises(ValidationError):
            Need(name="test", intensity=1.5)

        with pytest.raises(ValidationError):
            Need(name="test", intensity=-0.1)

    def test_need_update_intensity(self):
        """Test updating need intensity."""
        need = Need(name="test", intensity=0.5)

        # Increase intensity
        need.update_intensity(0.2)
        assert need.intensity == 0.7

        # Decrease intensity
        need.update_intensity(-0.3)
        assert need.intensity == 0.4

        # Cannot exceed bounds
        need.update_intensity(1.0)
        assert need.intensity == 1.0

        need.update_intensity(-2.0)
        assert need.intensity == 0.0

    def test_need_satisfy(self):
        """Test satisfying a need."""
        need = Need(name="test", intensity=0.8, satisfaction_level=0.2)

        # Satisfy the need
        need.satisfy(0.5)

        # Satisfaction should increase
        assert need.satisfaction_level > 0.2

        # Intensity should decrease
        assert need.intensity < 0.8

    def test_need_satisfy_bounds(self):
        """Test that satisfaction level stays within bounds."""
        need = Need(name="test", satisfaction_level=0.9)

        # Satisfying beyond 1.0 should cap at 1.0
        need.satisfy(0.5)
        assert need.satisfaction_level == 1.0


class TestSchemaIntegration:
    """Integration tests for schemas."""

    def test_memory_with_belief_relationship(self):
        """Test linking memories to beliefs."""
        memory = Memory(
            id="mem_123",
            content={"observation": "ball is round"},
            strength=1.0
        )

        belief = Belief(
            subject="ball",
            predicate="is",
            object="round",
            supporting_memories=[memory.id]
        )

        assert memory.id in belief.supporting_memories

    def test_developmental_stage_consistency(self):
        """Test that developmental stages are consistent across schemas."""
        stage = DevelopmentalStage.TODDLER

        memory = Memory(
            id="test",
            content={},
            developmental_stage=stage
        )
        belief = Belief(
            subject="test",
            predicate="is",
            object="test",
            developmental_stage=stage
        )
        message = NetworkMessage(
            sender="test",
            receiver="test",
            content={},
            developmental_stage=stage
        )

        assert memory.developmental_stage == stage
        assert belief.developmental_stage == stage
        assert message.developmental_stage == stage

    def test_schema_serialization(self):
        """Test that schemas can be serialized to dictionaries."""
        message = NetworkMessage(
            sender="test",
            receiver="target",
            content={"data": "value"}
        )
        state = NetworkState(name="test")

        # Test serialization
        msg_dict = message.to_dict()
        state_dict = state.to_dict()

        assert isinstance(msg_dict, dict)
        assert isinstance(state_dict, dict)
        assert "sender" in msg_dict
        assert "name" in state_dict
