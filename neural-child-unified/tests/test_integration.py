#----------------------------------------------------------------------------
#File:       test_integration.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Integration tests for the unified neural child system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Integration tests for the unified neural child system.

Run with: pytest tests/test_integration.py -v
Or: python tests/test_integration.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_core_imports():
    """Test that core modules can be imported."""
    try:
        from neural_child.models.schemas import DevelopmentalStage, NetworkMessage
        assert DevelopmentalStage is not None
        assert NetworkMessage is not None
        print("✓ Core schemas imported")
        return True
    except Exception as e:
        print(f"✗ Core schemas import failed: {e}")
        return False


def test_emotional_systems():
    """Test that emotional systems can be imported."""
    try:
        from neural_child.emotional.regulation import EmotionalRegulation
        from neural_child.emotional.development import EmotionalDevelopmentSystem
        from neural_child.emotional.memory import EmotionalMemorySystem
        print("✓ Emotional systems imported")
        return True
    except Exception as e:
        print(f"✗ Emotional systems import failed: {e}")
        return False


def test_interaction_systems():
    """Test that interaction systems can be imported."""
    try:
        from neural_child.interaction.llm.llm_module import chat_completion
        from neural_child.interaction.chat.integrated_chat import IntegratedChatSystem
        print("✓ Interaction systems imported")
        return True
    except Exception as e:
        print(f"✗ Interaction systems import failed: {e}")
        return False


def test_psychological_components():
    """Test that psychological components can be imported."""
    try:
        from neural_child.psychological.attachment import AttachmentSystem
        from neural_child.psychological.theory_of_mind import TheoryOfMind
        from neural_child.psychological.defense_mechanisms import DefenseMechanisms
        print("✓ Psychological components imported")
        return True
    except Exception as e:
        print(f"✗ Psychological components import failed: {e}")
        return False


def test_unique_features():
    """Test that unique features can be imported."""
    try:
        from neural_child.physiological.heartbeat_system import HeartbeatSystem
        from neural_child.communication.message_bus import MessageBus
        from neural_child.learning.autonomous_learner import AutonomousLearner
        from neural_child.safety.safety_monitor import SafetyMonitor
        print("✓ Unique features imported")
        return True
    except Exception as e:
        print(f"✗ Unique features import failed: {e}")
        return False


def test_dream_system():
    """Test that dream system can be imported."""
    try:
        from neural_child.dream.dream_generator import DreamGenerator
        from neural_child.dream.dream_system import DreamSystem, DreamContent, DreamType
        print("✓ Dream system imported")
        return True
    except Exception as e:
        print(f"✗ Dream system import failed: {e}")
        return False


def test_web_interface():
    """Test that web interface can be imported and created."""
    try:
        from neural_child.web.app import app_factory
        app = app_factory()
        assert app is not None
        print("✓ Web interface created")
        return True
    except Exception as e:
        print(f"✗ Web interface creation failed: {e}")
        return False


def test_visualization():
    """Test that visualization tools can be imported."""
    try:
        from neural_child.visualization.visualization import (
            EmotionalStateVisualizer,
            NeuralNetworkVisualizer
        )
        print("✓ Visualization tools imported")
        return True
    except Exception as e:
        print(f"✗ Visualization tools import failed: {e}")
        return False


def test_obsidian_integration():
    """Test that Obsidian integration can be imported."""
    try:
        from neural_child.integration.obsidian.obsidian_api import ObsidianAPI
        print("✓ Obsidian integration imported")
        return True
    except Exception as e:
        print(f"✗ Obsidian integration import failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Neural Child Development System - Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Emotional Systems", test_emotional_systems),
        ("Interaction Systems", test_interaction_systems),
        ("Psychological Components", test_psychological_components),
        ("Unique Features", test_unique_features),
        ("Dream System", test_dream_system),
        ("Web Interface", test_web_interface),
        ("Visualization", test_visualization),
        ("Obsidian Integration", test_obsidian_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test raised exception: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

