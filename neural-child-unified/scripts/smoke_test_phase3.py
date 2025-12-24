#!/usr/bin/env python3
"""
Smoke test for Phase 3: Cognitive Systems
Verifies that all cognitive system modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all Phase 3 imports."""
    errors = []
    
    print("Testing Phase 3: Cognitive Systems imports...")
    print("=" * 60)
    
    # Test memory systems
    print("\n1. Testing memory systems...")
    try:
        from neural_child.cognitive.memory.memory_context import MemoryContext
        print("  ✓ memory_context")
    except Exception as e:
        errors.append(f"memory_context: {e}")
        print(f"  ✗ memory_context: {e}")
    
    try:
        from neural_child.cognitive.memory.rag_memory import RAGMemory
        print("  ✓ rag_memory")
    except Exception as e:
        errors.append(f"rag_memory: {e}")
        print(f"  ✗ rag_memory: {e}")
    
    try:
        from neural_child.cognitive.memory.memory_store import MemoryStore
        print("  ✓ memory_store")
    except Exception as e:
        errors.append(f"memory_store: {e}")
        print(f"  ✗ memory_store: {e}")
    
    try:
        from neural_child.cognitive.memory.memory_module import DifferentiableMemory
        print("  ✓ memory_module")
    except Exception as e:
        errors.append(f"memory_module: {e}")
        print(f"  ✗ memory_module: {e}")
    
    # Test language systems
    print("\n2. Testing language systems...")
    try:
        from neural_child.cognitive.language.text_embed import get_embeddings, initialize_embedding_model
        print("  ✓ text_embed")
    except Exception as e:
        errors.append(f"text_embed: {e}")
        print(f"  ✗ text_embed: {e}")
    
    try:
        from neural_child.cognitive.language.symbol_grounding import SymbolGrounding
        print("  ✓ symbol_grounding")
    except Exception as e:
        errors.append(f"symbol_grounding: {e}")
        print(f"  ✗ symbol_grounding: {e}")
    
    try:
        from neural_child.cognitive.language.language_development import (
            LanguageDevelopment, LanguageStage, WordCategory
        )
        print("  ✓ language_development")
    except Exception as e:
        errors.append(f"language_development: {e}")
        print(f"  ✗ language_development: {e}")
    
    # Test vision systems
    print("\n3. Testing vision systems...")
    try:
        from neural_child.cognitive.vision.vision_development import VisionDevelopment
        print("  ✓ vision_development")
    except Exception as e:
        errors.append(f"vision_development: {e}")
        print(f"  ✗ vision_development: {e}")
    
    # Test metacognition
    print("\n4. Testing metacognition...")
    try:
        from neural_child.cognitive.metacognition.metacognition_system import (
            MetacognitionSystem, SelfAwarenessLevel
        )
        print("  ✓ metacognition_system")
    except Exception as e:
        errors.append(f"metacognition_system: {e}")
        print(f"  ✗ metacognition_system: {e}")
    
    # Test moral network
    print("\n5. Testing moral network...")
    try:
        from neural_child.cognitive.moral.moral_network import MoralNetwork, MoralValue
        print("  ✓ moral_network")
    except Exception as e:
        errors.append(f"moral_network: {e}")
        print(f"  ✗ moral_network: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"\n❌ FAILED: {len(errors)} import error(s)")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ SUCCESS: All Phase 3 cognitive systems imported successfully!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

