#----------------------------------------------------------------------------
#File:       smoke_test_phase1.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Smoke test for Phase 1 foundation
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Smoke test for Phase 1 foundation.

Tests that all Phase 1 components can be imported and basic structure is in place.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all Phase 1 modules can be imported."""
    print("Testing Phase 1 imports...")
    
    try:
        # Test package import
        import neural_child
        print("✅ neural_child package imported")
        
        # Test schemas
        from neural_child.models.schemas import (
            DevelopmentalStage,
            NetworkMessage,
            Memory,
            Belief,
            Need,
            MotherResponse,
            EmotionalContext,
            ActionType
        )
        print("✅ All schemas imported")
        
        # Test utilities
        from neural_child.utils.logger import DevelopmentLogger
        from neural_child.utils.helpers import (
            parse_llm_response,
            ensure_tensor,
            format_time_delta
        )
        print("✅ Utilities imported")
        
        # Test version
        from neural_child.version import __version__
        print(f"✅ Version: {__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_schemas():
    """Test that schemas work correctly."""
    print("\nTesting schemas...")
    
    try:
        from neural_child.models.schemas import (
            DevelopmentalStage,
            NetworkMessage,
            Memory,
            MotherResponse,
            EmotionalContext
        )
        
        # Test DevelopmentalStage
        stage = DevelopmentalStage.INFANT
        assert stage.value == 1
        print("✅ DevelopmentalStage works")
        
        # Test NetworkMessage
        msg = NetworkMessage(
            sender="test_sender",
            receiver="test_receiver",
            content={"test": "data"}
        )
        assert msg.sender == "test_sender"
        print("✅ NetworkMessage works")
        
        # Test Memory
        memory = Memory(
            id="test_memory",
            content={"test": "data"}
        )
        assert memory.id == "test_memory"
        print("✅ Memory works")
        
        # Test MotherResponse
        response = MotherResponse(
            content="Test response",
            emotional_context=EmotionalContext()
        )
        assert response.content == "Test response"
        print("✅ MotherResponse works")
        
        return True
    except Exception as e:
        print(f"❌ Schema test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that directory structure is correct."""
    print("\nTesting directory structure...")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src/neural_child",
        "src/neural_child/core",
        "src/neural_child/core/brain",
        "src/neural_child/core/decision",
        "src/neural_child/core/development",
        "src/neural_child/core/training",
        "src/neural_child/cognitive",
        "src/neural_child/cognitive/memory",
        "src/neural_child/cognitive/language",
        "src/neural_child/emotional",
        "src/neural_child/interaction",
        "src/neural_child/models",
        "src/neural_child/utils",
        "config",
        "tests",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            all_exist = False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    return all_exist


def test_config_file():
    """Test that config file exists."""
    print("\nTesting config file...")
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if config_path.exists():
        print("✅ config/config.yaml exists")
        return True
    else:
        print("❌ config/config.yaml not found")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Phase 1 Smoke Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Schemas", test_schemas()))
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Config File", test_config_file()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All Phase 1 smoke tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

