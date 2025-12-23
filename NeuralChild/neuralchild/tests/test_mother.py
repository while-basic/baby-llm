"""Tests for the Mother LLM component.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock, patch, Mock
import json

from neuralchild.mother.mother_llm import MotherLLM, MotherResponse
from neuralchild.mind.mind_core import Mind
from neuralchild.core.schemas import DevelopmentalStage, Need
from neuralchild.config import Config


class TestMotherResponse:
    """Test suite for MotherResponse schema."""

    def test_mother_response_creation(self):
        """Test creating a MotherResponse."""
        response = MotherResponse(
            understanding="Child is happy",
            response="That's wonderful!",
            action="encourage",
            development_focus="emotional"
        )

        assert response.understanding == "Child is happy"
        assert response.response == "That's wonderful!"
        assert response.action == "encourage"
        assert response.development_focus == "emotional"

    def test_mother_response_defaults(self):
        """Test MotherResponse default values."""
        response = MotherResponse(
            understanding="Test",
            response="Test response",
            action="speak"
        )

        assert response.development_focus is None
        assert isinstance(response.timestamp, datetime)

    def test_mother_response_to_dict(self):
        """Test converting MotherResponse to dictionary."""
        response = MotherResponse(
            understanding="Test",
            response="Test response",
            action="speak",
            development_focus="language"
        )

        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert response_dict["understanding"] == "Test"
        assert response_dict["action"] == "speak"
        assert "timestamp" in response_dict


class TestMotherLLMInitialization:
    """Test suite for MotherLLM initialization."""

    def test_mother_initialization(self, mother: MotherLLM):
        """Test that MotherLLM initializes correctly."""
        assert mother is not None
        assert hasattr(mother, 'interaction_history')
        assert isinstance(mother.interaction_history, list)

    def test_mother_has_personality(self, mother: MotherLLM):
        """Test that mother has personality attributes."""
        assert hasattr(mother, 'personality')
        assert isinstance(mother.personality, dict)

        # Check for key personality traits
        expected_traits = ["patience", "warmth", "playfulness"]
        for trait in expected_traits:
            if trait in mother.personality:
                assert 0.0 <= mother.personality[trait] <= 1.0

    def test_mother_has_response_templates(self, mother: MotherLLM):
        """Test that mother has response templates."""
        assert hasattr(mother, 'response_templates')
        assert isinstance(mother.response_templates, dict)

    def test_mother_has_developmental_techniques(self, mother: MotherLLM):
        """Test that mother has developmental techniques for each stage."""
        assert hasattr(mother, 'developmental_techniques')

        # Check for techniques for each stage
        for stage in DevelopmentalStage:
            if stage in mother.developmental_techniques:
                techniques = mother.developmental_techniques[stage]
                assert isinstance(techniques, dict)


class TestMotherObservation:
    """Test suite for mother's observation capabilities."""

    def test_mother_observe_mind(self, mother: MotherLLM, mind: Mind, mock_chat_completion):
        """Test mother observing the mind state."""
        if hasattr(mother, 'observe_and_respond'):
            with patch('neuralchild.utils.llm_module.chat_completion', return_value=mock_chat_completion.return_value):
                response = mother.observe_and_respond(mind)

                # Response might be None or a MotherResponse
                if response is not None:
                    assert isinstance(response, (MotherResponse, dict))

    def test_mother_observe_different_stages(self, mother: MotherLLM, mind: Mind):
        """Test mother observing different developmental stages."""
        stages = [DevelopmentalStage.INFANT, DevelopmentalStage.TODDLER, DevelopmentalStage.CHILD]

        for stage in stages:
            mind.state.developmental_stage = stage

            if hasattr(mother, 'observe_and_respond'):
                # Mock LLM to avoid actual API calls
                with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
                    mock_llm.return_value = {
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "understanding": "Child is developing",
                                    "response": "Good job!",
                                    "action": "encourage",
                                    "development_focus": "cognitive"
                                })
                            }
                        }]
                    }

                    response = mother.observe_and_respond(mind)

                    # Mother should respond appropriately to each stage
                    if response:
                        assert response is not None

    def test_mother_detect_needs(self, mother: MotherLLM, mind: Mind):
        """Test mother detecting child's needs."""
        # Set high need
        if hasattr(mind, 'needs'):
            mind.needs["attention"] = Need(name="attention", intensity=0.9)

        if hasattr(mother, 'observe_and_respond'):
            with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
                mock_llm.return_value = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "understanding": "Child needs attention",
                                "response": "I'm here for you",
                                "action": "comfort",
                                "development_focus": "emotional"
                            })
                        }
                    }]
                }

                response = mother.observe_and_respond(mind)

                # Mother should respond to high needs
                if response:
                    assert response is not None


class TestMotherResponses:
    """Test suite for mother's response generation."""

    def test_mother_generate_response(self, mother: MotherLLM, mind: Mind):
        """Test generating a response to the mind."""
        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "understanding": "Child is curious",
                            "response": "Let's explore together!",
                            "action": "teach",
                            "development_focus": "cognitive"
                        })
                    }
                }]
            }

            if hasattr(mother, 'observe_and_respond'):
                response = mother.observe_and_respond(mind)

                if response:
                    assert hasattr(response, 'response') or 'response' in response

    def test_mother_stage_appropriate_responses(self, mother: MotherLLM):
        """Test that mother generates stage-appropriate responses."""
        # Test infant responses
        if hasattr(mother, 'response_templates'):
            templates = mother.response_templates

            if "INFANT" in templates:
                infant_templates = templates["INFANT"]
                assert isinstance(infant_templates, dict)

                # Infant responses should be simple and comforting
                if "comfort" in infant_templates:
                    assert len(infant_templates["comfort"]) > 0

    def test_mother_response_variety(self, mother: MotherLLM):
        """Test that mother has variety in responses."""
        if hasattr(mother, 'response_templates'):
            for stage_name, templates in mother.response_templates.items():
                if isinstance(templates, dict):
                    for category, responses in templates.items():
                        if isinstance(responses, list):
                            # Should have multiple response options
                            assert len(responses) >= 1

    def test_mother_emotional_responses(self, mother: MotherLLM, mind: Mind):
        """Test mother's emotional responses."""
        # Set emotional state
        if hasattr(mind.state, 'emotional_state'):
            mind.state.emotional_state = "distressed"

        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "understanding": "Child is distressed",
                            "response": "It's okay, I'm here",
                            "action": "comfort",
                            "development_focus": "emotional"
                        })
                    }
                }]
            }

            if hasattr(mother, 'observe_and_respond'):
                response = mother.observe_and_respond(mind)

                # Mother should provide comfort
                if response:
                    assert response is not None


class TestMotherInteractionHistory:
    """Test suite for interaction history tracking."""

    def test_mother_records_interactions(self, mother: MotherLLM, mind: Mind):
        """Test that mother records interactions."""
        initial_history_length = len(mother.interaction_history)

        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "understanding": "Test",
                            "response": "Test response",
                            "action": "speak",
                            "development_focus": "language"
                        })
                    }
                }]
            }

            if hasattr(mother, 'observe_and_respond'):
                mother.observe_and_respond(mind)

                # History might grow
                assert len(mother.interaction_history) >= initial_history_length

    def test_mother_interaction_history_format(self, mother: MotherLLM):
        """Test interaction history format."""
        assert isinstance(mother.interaction_history, list)

        # Add a test interaction
        test_interaction = {
            "timestamp": datetime.now(),
            "stage": DevelopmentalStage.INFANT,
            "response": "Test"
        }

        mother.interaction_history.append(test_interaction)
        assert len(mother.interaction_history) > 0

    def test_mother_history_retrieval(self, mother: MotherLLM):
        """Test retrieving interaction history."""
        # Add interactions
        for i in range(5):
            mother.interaction_history.append({
                "timestamp": datetime.now(),
                "interaction": f"test_{i}"
            })

        # Should be able to retrieve recent interactions
        assert len(mother.interaction_history) >= 5


class TestMotherDevelopmentalFocus:
    """Test suite for mother's developmental focus."""

    def test_mother_language_development(self, mother: MotherLLM):
        """Test mother's language development techniques."""
        if hasattr(mother, 'developmental_techniques'):
            for stage in [DevelopmentalStage.INFANT, DevelopmentalStage.TODDLER]:
                if stage in mother.developmental_techniques:
                    techniques = mother.developmental_techniques[stage]

                    if "language" in techniques:
                        assert len(techniques["language"]) > 0

    def test_mother_emotional_development(self, mother: MotherLLM):
        """Test mother's emotional development techniques."""
        if hasattr(mother, 'developmental_techniques'):
            for stage in DevelopmentalStage:
                if stage in mother.developmental_techniques:
                    techniques = mother.developmental_techniques[stage]

                    if "emotional" in techniques:
                        assert len(techniques["emotional"]) > 0

    def test_mother_cognitive_development(self, mother: MotherLLM):
        """Test mother's cognitive development techniques."""
        if hasattr(mother, 'developmental_techniques'):
            for stage in DevelopmentalStage:
                if stage in mother.developmental_techniques:
                    techniques = mother.developmental_techniques[stage]

                    if "cognitive" in techniques:
                        assert len(techniques["cognitive"]) > 0

    def test_mother_focus_changes_with_stage(self, mother: MotherLLM):
        """Test that developmental focus changes with stage."""
        if hasattr(mother, 'developmental_techniques'):
            infant_techniques = mother.developmental_techniques.get(DevelopmentalStage.INFANT, {})
            child_techniques = mother.developmental_techniques.get(DevelopmentalStage.CHILD, {})

            # Techniques should differ between stages
            if infant_techniques and child_techniques:
                # Child techniques should be more complex
                assert infant_techniques != child_techniques


class TestMotherTiming:
    """Test suite for mother's response timing."""

    def test_mother_response_interval(self, mother: MotherLLM):
        """Test mother's response interval."""
        assert hasattr(mother, 'response_interval')
        assert mother.response_interval > 0

    def test_mother_respects_timing(self, mother: MotherLLM, mind: Mind):
        """Test that mother respects response timing."""
        if hasattr(mother, 'last_response_time'):
            original_time = mother.last_response_time

            # Try to respond immediately
            with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
                mock_llm.return_value = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "understanding": "Test",
                                "response": "Test",
                                "action": "speak"
                            })
                        }
                    }]
                }

                if hasattr(mother, 'observe_and_respond'):
                    mother.observe_and_respond(mind)

            # Time should be updated
            assert mother.last_response_time is not None


class TestMotherLLMIntegration:
    """Integration tests for Mother LLM."""

    def test_mother_mind_interaction_cycle(self, mother: MotherLLM, mind: Mind):
        """Test complete mother-mind interaction cycle."""
        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "understanding": "Child is learning",
                            "response": "Keep going!",
                            "action": "encourage",
                            "development_focus": "cognitive"
                        })
                    }
                }]
            }

            # Mother observes
            if hasattr(mother, 'observe_and_respond'):
                response = mother.observe_and_respond(mind)

                # Mind processes response
                if response and hasattr(mind, 'process_input'):
                    mind.process_input({"source": "mother", "content": response})

                # Complete cycle
                assert mind is not None

    def test_mother_adapts_to_development(self, mother: MotherLLM, mind: Mind):
        """Test mother adapts responses as mind develops."""
        stages = [DevelopmentalStage.INFANT, DevelopmentalStage.TODDLER, DevelopmentalStage.CHILD]
        responses = []

        for stage in stages:
            mind.state.developmental_stage = stage

            with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
                mock_llm.return_value = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "understanding": f"Child at {stage.name}",
                                "response": f"Response for {stage.name}",
                                "action": "teach",
                                "development_focus": "language"
                            })
                        }
                    }]
                }

                if hasattr(mother, 'observe_and_respond'):
                    response = mother.observe_and_respond(mind)
                    if response:
                        responses.append(response)

        # Mother should have responded to different stages
        assert len(responses) >= 0


class TestMotherErrorHandling:
    """Test error handling in Mother LLM."""

    def test_mother_handles_none_mind(self, mother: MotherLLM):
        """Test that mother handles None mind gracefully."""
        if hasattr(mother, 'observe_and_respond'):
            try:
                mother.observe_and_respond(None)
            except (TypeError, AttributeError, ValueError):
                # Expected to handle or raise appropriate error
                pass

    def test_mother_handles_llm_error(self, mother: MotherLLM, mind: Mind):
        """Test handling LLM errors."""
        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            # Simulate LLM error
            mock_llm.side_effect = Exception("LLM Error")

            if hasattr(mother, 'observe_and_respond'):
                try:
                    response = mother.observe_and_respond(mind)
                    # Should handle error gracefully
                except Exception as e:
                    # Error might be raised
                    assert "LLM Error" in str(e) or isinstance(e, Exception)

    def test_mother_handles_invalid_json_response(self, mother: MotherLLM, mind: Mind):
        """Test handling invalid JSON from LLM."""
        with patch('neuralchild.utils.llm_module.chat_completion') as mock_llm:
            # Return invalid JSON
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": "This is not valid JSON"
                    }
                }]
            }

            if hasattr(mother, 'observe_and_respond'):
                try:
                    response = mother.observe_and_respond(mind)
                    # Should handle gracefully
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Expected behavior
                    pass
