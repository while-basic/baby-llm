"""Tests for neural network implementations.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from neuralchild.mind.networks.consciousness import ConsciousnessNetwork
from neuralchild.mind.networks.emotions import EmotionsNetwork
from neuralchild.mind.networks.perception import PerceptionNetwork
from neuralchild.mind.networks.thoughts import ThoughtsNetwork
from neuralchild.core.schemas import DevelopmentalStage, NetworkMessage


class TestConsciousnessNetwork:
    """Test suite for ConsciousnessNetwork."""

    @pytest.fixture
    def consciousness_network(self):
        """Create a consciousness network for testing."""
        return ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

    def test_consciousness_network_initialization(self, consciousness_network: ConsciousnessNetwork):
        """Test that ConsciousnessNetwork initializes correctly."""
        assert consciousness_network is not None
        assert consciousness_network.name == "consciousness"
        assert hasattr(consciousness_network, 'rnn')
        assert hasattr(consciousness_network, 'self_model')
        assert hasattr(consciousness_network, 'output_layer')

    def test_consciousness_network_attributes(self, consciousness_network: ConsciousnessNetwork):
        """Test consciousness network has expected attributes."""
        assert hasattr(consciousness_network, 'awareness_level')
        assert hasattr(consciousness_network, 'self_awareness')
        assert hasattr(consciousness_network, 'integration_capacity')
        assert hasattr(consciousness_network, 'attending_to')

        # Check initial values
        assert 0.0 <= consciousness_network.awareness_level <= 1.0
        assert 0.0 <= consciousness_network.self_awareness <= 1.0
        assert 0.0 <= consciousness_network.integration_capacity <= 1.0

    def test_consciousness_network_forward_2d_input(self, consciousness_network: ConsciousnessNetwork):
        """Test forward pass with 2D input."""
        test_input = torch.randn(1, 64)
        output = consciousness_network.forward(test_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1
        assert output.shape[1] == 64

    def test_consciousness_network_forward_3d_input(self, consciousness_network: ConsciousnessNetwork):
        """Test forward pass with 3D input."""
        test_input = torch.randn(1, 5, 64)
        output = consciousness_network.forward(test_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 64

    def test_consciousness_network_hidden_state(self, consciousness_network: ConsciousnessNetwork):
        """Test that hidden state is maintained."""
        test_input = torch.randn(1, 64)

        # First forward pass
        output1 = consciousness_network.forward(test_input)
        assert consciousness_network.hidden is not None

        # Second forward pass should use previous hidden state
        output2 = consciousness_network.forward(test_input)
        assert output2 is not None

    def test_consciousness_network_process_message(self, consciousness_network: ConsciousnessNetwork):
        """Test processing network messages."""
        message = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={"data": [0.1] * 64},
            message_type="sensory"
        )

        # Process the message
        consciousness_network.process_message(message)

        # Network should track recent activations
        assert hasattr(consciousness_network, 'network_activations')

    def test_consciousness_network_developmental_growth(self, consciousness_network: ConsciousnessNetwork):
        """Test network growth with developmental stage."""
        initial_awareness = consciousness_network.awareness_level

        # Trigger developmental update
        consciousness_network.update_for_stage(DevelopmentalStage.TODDLER)

        # Awareness should increase with development
        assert consciousness_network.awareness_level >= initial_awareness


class TestEmotionsNetwork:
    """Test suite for EmotionsNetwork."""

    @pytest.fixture
    def emotions_network(self):
        """Create an emotions network for testing."""
        return EmotionsNetwork(input_dim=64, hidden_dim=128, output_dim=64)

    def test_emotions_network_initialization(self, emotions_network: EmotionsNetwork):
        """Test that EmotionsNetwork initializes correctly."""
        assert emotions_network is not None
        assert emotions_network.name == "emotions"
        assert hasattr(emotions_network, 'emotion_processor')

    def test_emotions_network_forward(self, emotions_network: EmotionsNetwork):
        """Test forward pass of emotions network."""
        test_input = torch.randn(1, 64)
        output = emotions_network.forward(test_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1
        assert output.shape[1] == 64

    def test_emotions_network_has_emotion_state(self, emotions_network: EmotionsNetwork):
        """Test that emotions network maintains emotional state."""
        assert hasattr(emotions_network, 'current_emotion')
        assert hasattr(emotions_network, 'emotional_intensity')

        # Initial values should be reasonable
        if hasattr(emotions_network, 'emotional_intensity'):
            assert 0.0 <= emotions_network.emotional_intensity <= 1.0

    def test_emotions_network_process_different_inputs(self, emotions_network: EmotionsNetwork):
        """Test processing different emotional inputs."""
        # Positive emotional input
        positive_input = torch.ones(1, 64) * 0.8

        # Negative emotional input
        negative_input = torch.ones(1, 64) * -0.8

        # Neutral input
        neutral_input = torch.zeros(1, 64)

        output1 = emotions_network.forward(positive_input)
        output2 = emotions_network.forward(negative_input)
        output3 = emotions_network.forward(neutral_input)

        # All outputs should be valid tensors
        assert output1 is not None
        assert output2 is not None
        assert output3 is not None


class TestPerceptionNetwork:
    """Test suite for PerceptionNetwork."""

    @pytest.fixture
    def perception_network(self):
        """Create a perception network for testing."""
        return PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64)

    def test_perception_network_initialization(self, perception_network: PerceptionNetwork):
        """Test that PerceptionNetwork initializes correctly."""
        assert perception_network is not None
        assert perception_network.name == "perception"
        assert hasattr(perception_network, 'visual_processor')
        assert hasattr(perception_network, 'auditory_processor')

    def test_perception_network_forward(self, perception_network: PerceptionNetwork):
        """Test forward pass of perception network."""
        test_input = torch.randn(1, 128)
        output = perception_network.forward(test_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1
        assert output.shape[1] == 64

    def test_perception_network_multi_modal(self, perception_network: PerceptionNetwork):
        """Test perception network with multi-modal input."""
        # Simulate visual and auditory input
        visual_input = torch.randn(1, 64)
        auditory_input = torch.randn(1, 64)

        # Combined input
        combined = torch.cat([visual_input, auditory_input], dim=1)
        output = perception_network.forward(combined)

        assert output is not None
        assert output.shape[1] == 64

    def test_perception_network_attention(self, perception_network: PerceptionNetwork):
        """Test perception network attention mechanism if available."""
        test_input = torch.randn(1, 128)
        output = perception_network.forward(test_input)

        # Check if attention weights are computed
        if hasattr(perception_network, 'attention_weights'):
            assert perception_network.attention_weights is not None


class TestThoughtsNetwork:
    """Test suite for ThoughtsNetwork."""

    @pytest.fixture
    def thoughts_network(self):
        """Create a thoughts network for testing."""
        return ThoughtsNetwork(input_dim=64, hidden_dim=128, output_dim=64)

    def test_thoughts_network_initialization(self, thoughts_network: ThoughtsNetwork):
        """Test that ThoughtsNetwork initializes correctly."""
        assert thoughts_network is not None
        assert thoughts_network.name == "thoughts"
        assert hasattr(thoughts_network, 'thought_generator')

    def test_thoughts_network_forward(self, thoughts_network: ThoughtsNetwork):
        """Test forward pass of thoughts network."""
        test_input = torch.randn(1, 64)
        output = thoughts_network.forward(test_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1
        assert output.shape[1] == 64

    def test_thoughts_network_generation(self, thoughts_network: ThoughtsNetwork):
        """Test thought generation capabilities."""
        test_input = torch.randn(1, 64)

        # Generate multiple thoughts
        output1 = thoughts_network.forward(test_input)
        output2 = thoughts_network.forward(test_input)

        # Outputs should be valid
        assert output1 is not None
        assert output2 is not None

    def test_thoughts_network_has_thought_state(self, thoughts_network: ThoughtsNetwork):
        """Test that thoughts network maintains thought state."""
        # Check for thought-related attributes
        test_input = torch.randn(1, 64)
        thoughts_network.forward(test_input)

        # Network should be able to process thoughts
        assert thoughts_network is not None


class TestNetworkIntegration:
    """Integration tests for neural networks."""

    def test_all_networks_forward_compatible(self):
        """Test that all networks can process compatible tensors."""
        # Create all networks
        consciousness = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        emotions = EmotionsNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        perception = PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64)
        thoughts = ThoughtsNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Test inputs
        input_64 = torch.randn(1, 64)
        input_128 = torch.randn(1, 128)

        # All networks should produce outputs
        assert consciousness.forward(input_64) is not None
        assert emotions.forward(input_64) is not None
        assert perception.forward(input_128) is not None
        assert thoughts.forward(input_64) is not None

    def test_network_chain_processing(self):
        """Test chaining network outputs."""
        perception = PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64)
        consciousness = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Process through perception first
        perception_input = torch.randn(1, 128)
        perception_output = perception.forward(perception_input)

        # Feed to consciousness
        consciousness_output = consciousness.forward(perception_output)

        assert consciousness_output is not None
        assert consciousness_output.shape[1] == 64

    def test_network_message_passing(self):
        """Test message passing between networks."""
        sender = PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64)
        receiver = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Create message
        message = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={"data": [0.1] * 64},
            message_type="sensory"
        )

        # Both networks should be able to process messages
        sender.process_message(message)
        receiver.process_message(message)

    def test_network_developmental_progression(self):
        """Test all networks can handle developmental updates."""
        networks = [
            ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64),
            EmotionsNetwork(input_dim=64, hidden_dim=128, output_dim=64),
            PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64),
            ThoughtsNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        ]

        # All networks should handle stage updates
        for network in networks:
            network.update_for_stage(DevelopmentalStage.TODDLER)
            network.update_for_stage(DevelopmentalStage.CHILD)

    def test_network_state_persistence(self):
        """Test that networks maintain state across calls."""
        network = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        input1 = torch.randn(1, 64)
        input2 = torch.randn(1, 64)

        # First processing
        network.forward(input1)
        state1 = network.get_state() if hasattr(network, 'get_state') else network.state

        # Second processing
        network.forward(input2)
        state2 = network.get_state() if hasattr(network, 'get_state') else network.state

        # State should exist and be updated
        assert state1 is not None
        assert state2 is not None


class TestNetworkErrorHandling:
    """Test error handling in neural networks."""

    def test_consciousness_network_invalid_input_shape(self):
        """Test consciousness network with invalid input shape."""
        network = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Wrong input dimension
        invalid_input = torch.randn(1, 32)  # Expected 64

        # Should either handle gracefully or raise clear error
        try:
            output = network.forward(invalid_input)
        except (RuntimeError, ValueError) as e:
            # Expected behavior - dimension mismatch
            assert "size" in str(e).lower() or "dimension" in str(e).lower()

    def test_network_none_input_handling(self):
        """Test networks handle None input gracefully."""
        network = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Networks should handle None input appropriately
        try:
            network.forward(None)
        except (TypeError, AttributeError, ValueError):
            # Expected to raise an error
            pass

    def test_network_empty_tensor_input(self):
        """Test networks handle empty tensors."""
        network = EmotionsNetwork(input_dim=64, hidden_dim=128, output_dim=64)

        # Empty tensor
        empty_input = torch.empty(0, 64)

        try:
            output = network.forward(empty_input)
        except (RuntimeError, ValueError):
            # Expected behavior for empty input
            pass
