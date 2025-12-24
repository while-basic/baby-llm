#----------------------------------------------------------------------------
#File:       smoke_test_phase4.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Smoke test for Phase 4 - Emotional & Interaction Systems
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Smoke test for Phase 4 - Emotional & Interaction Systems.

This test verifies that all Phase 4 modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all Phase 4 modules can be imported."""
    errors = []
    successes = []

    # Test Phase 4.1: Emotional Systems
    print("Testing Phase 4.1: Emotional Systems...")
    try:
        from neural_child.emotional.regulation import (
            EmotionalRegulation,
            EmotionalState
        )
        successes.append("✅ emotional/regulation.py")
    except Exception as e:
        errors.append(f"❌ emotional/regulation.py: {e}")

    try:
        from neural_child.emotional.development import (
            EmotionalDevelopmentSystem,
            EmotionalCapability,
            EmotionalState as DevelopmentEmotionalState
        )
        successes.append("✅ emotional/development.py")
    except Exception as e:
        errors.append(f"❌ emotional/development.py: {e}")

    try:
        from neural_child.emotional.memory import (
            EmotionalMemorySystem,
            EmotionalMemoryEntry,
            EmotionalAssociation
        )
        successes.append("✅ emotional/memory.py")
    except Exception as e:
        errors.append(f"❌ emotional/memory.py: {e}")

    try:
        from neural_child.emotional.embedding import (
            EmotionalEmbedder,
            QuantumEmotionalProcessor
        )
        successes.append("✅ emotional/embedding.py")
    except Exception as e:
        errors.append(f"❌ emotional/embedding.py: {e}")

    # Test Phase 4.2: Chat Systems
    print("\nTesting Phase 4.2: Chat Systems...")
    try:
        from neural_child.interaction.chat.integrated_chat import IntegratedChatSystem
        successes.append("✅ interaction/chat/integrated_chat.py")
    except Exception as e:
        errors.append(f"❌ interaction/chat/integrated_chat.py: {e}")

    try:
        from neural_child.interaction.chat.emotional_chat import EmotionalChatSystem
        successes.append("✅ interaction/chat/emotional_chat.py")
    except Exception as e:
        errors.append(f"❌ interaction/chat/emotional_chat.py: {e}")

    try:
        from neural_child.interaction.chat.self_awareness_chat import SelfAwarenessChatInterface
        successes.append("✅ interaction/chat/self_awareness_chat.py")
    except Exception as e:
        errors.append(f"❌ interaction/chat/self_awareness_chat.py: {e}")

    # Test Phase 4.3: LLM Integration
    print("\nTesting Phase 4.3: LLM Integration...")
    try:
        from neural_child.interaction.llm.llm_module import chat_completion
        successes.append("✅ interaction/llm/llm_module.py")
    except Exception as e:
        errors.append(f"❌ interaction/llm/llm_module.py: {e}")

    try:
        from neural_child.interaction.llm.ollama_chat import (
            OllamaChat,
            OllamaChildChat,
            get_child_response,
            analyze_sentiment
        )
        successes.append("✅ interaction/llm/ollama_chat.py")
    except Exception as e:
        errors.append(f"❌ interaction/llm/ollama_chat.py: {e}")

    # Test Phase 4.4: Mother LLM
    print("\nTesting Phase 4.4: Mother LLM...")
    try:
        from neural_child.interaction.llm.mother_llm import (
            MotherLLM,
            MotherResponse
        )
        successes.append("✅ interaction/llm/mother_llm.py")
    except Exception as e:
        errors.append(f"❌ interaction/llm/mother_llm.py: {e}")

    # Test Phase 4.5: Psychological Components
    print("\nTesting Phase 4.5: Psychological Components...")
    try:
        from neural_child.psychological.attachment import AttachmentSystem
        successes.append("✅ psychological/attachment.py")
    except Exception as e:
        errors.append(f"❌ psychological/attachment.py: {e}")

    try:
        from neural_child.psychological.theory_of_mind import TheoryOfMind
        successes.append("✅ psychological/theory_of_mind.py")
    except Exception as e:
        errors.append(f"❌ psychological/theory_of_mind.py: {e}")

    try:
        from neural_child.psychological.defense_mechanisms import DefenseMechanisms
        successes.append("✅ psychological/defense_mechanisms.py")
    except Exception as e:
        errors.append(f"❌ psychological/defense_mechanisms.py: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 4 SMOKE TEST RESULTS")
    print("=" * 60)
    
    if successes:
        print(f"\n✅ Successfully imported {len(successes)} modules:")
        for success in successes:
            print(f"  {success}")
    
    if errors:
        print(f"\n❌ Failed to import {len(errors)} modules:")
        for error in errors:
            print(f"  {error}")
    
    print("\n" + "=" * 60)
    
    if errors:
        print(f"❌ TEST FAILED: {len(errors)} import error(s)")
        return False
    else:
        print(f"✅ TEST PASSED: All {len(successes)} modules imported successfully!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

