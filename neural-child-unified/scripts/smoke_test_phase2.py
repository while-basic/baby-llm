#----------------------------------------------------------------------------
#File:       smoke_test_phase2.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Smoke test for Phase 2 - Core Systems
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Smoke test for Phase 2 - Core Systems.

This test verifies that all Phase 2 modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all Phase 2 modules can be imported."""
    errors = []
    successes = []

    # Test Phase 2.1: Brain Architecture
    print("Testing Phase 2.1: Brain Architecture...")
    try:
        from neural_child.core.brain.integrated_brain import (
            IntegratedBrain,
            BrainState,
            DevelopmentalStage as BrainDevStage
        )
        successes.append("✅ integrated_brain.py")
    except Exception as e:
        errors.append(f"❌ integrated_brain.py: {e}")

    try:
        from neural_child.core.brain.neural_architecture import (
            NeuralArchitecture,
            BrainRegion,
            CognitiveFunction
        )
        successes.append("✅ neural_architecture.py")
    except Exception as e:
        errors.append(f"❌ neural_architecture.py: {e}")

    # Test Phase 2.2: Decision Systems
    print("Testing Phase 2.2: Decision Systems...")
    try:
        from neural_child.core.decision.decision_network import (
            DecisionNetwork,
            ConversationEncoder
        )
        successes.append("✅ decision_network.py")
    except Exception as e:
        errors.append(f"❌ decision_network.py: {e}")

    try:
        from neural_child.core.decision.q_learning import (
            QLearningSystem,
            QNetwork
        )
        successes.append("✅ q_learning.py")
    except Exception as e:
        errors.append(f"❌ q_learning.py: {e}")

    # Test Phase 2.3: Developmental Stages
    print("Testing Phase 2.3: Developmental Stages...")
    try:
        from neural_child.development.stages import DevelopmentalStage
        successes.append("✅ stages.py")
    except Exception as e:
        errors.append(f"❌ stages.py: {e}")

    try:
        from neural_child.development.milestone_tracker import (
            MilestoneTracker,
            Milestone,
            DomainType
        )
        successes.append("✅ milestone_tracker.py")
    except Exception as e:
        errors.append(f"❌ milestone_tracker.py: {e}")

    try:
        from neural_child.development.curriculum_manager import (
            CurriculumManager
        )
        successes.append("✅ curriculum_manager.py")
    except Exception as e:
        errors.append(f"❌ curriculum_manager.py: {e}")

    # Test Phase 2.4: Training Systems
    print("Testing Phase 2.4: Training Systems...")
    try:
        from neural_child.core.training.self_supervised_trainer import (
            SelfSupervisedTrainer
        )
        successes.append("✅ self_supervised_trainer.py")
    except Exception as e:
        errors.append(f"❌ self_supervised_trainer.py: {e}")

    try:
        from neural_child.core.training.training_system import (
            MovingAverageMonitor,
            CheckpointManager,
            EarlyStopping
        )
        successes.append("✅ training_system.py")
    except Exception as e:
        errors.append(f"❌ training_system.py: {e}")

    try:
        from neural_child.core.training.replay_system import (
            ReplayOptimizer
        )
        successes.append("✅ replay_system.py")
    except Exception as e:
        errors.append(f"❌ replay_system.py: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"\n✅ Successes ({len(successes)}):")
    for success in successes:
        print(f"  {success}")

    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n🎉 All Phase 2 modules imported successfully!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

