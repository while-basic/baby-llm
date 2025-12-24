"""Dream systems for neural child development."""

# Import dream components
try:
    from neural_child.dream.dream_generator import DreamGenerator
    DREAM_GENERATOR_AVAILABLE = True
except ImportError:
    DreamGenerator = None
    DREAM_GENERATOR_AVAILABLE = False
    print("Warning: DreamGenerator not available.")

try:
    from neural_child.dream.dream_system import (
        DreamSystem,
        DreamContent,
        DreamType
    )
    DREAM_SYSTEM_AVAILABLE = True
except ImportError:
    DreamSystem = None
    DreamContent = None
    DreamType = None
    DREAM_SYSTEM_AVAILABLE = False
    print("Warning: DreamSystem not available.")

__all__ = [
    'DreamGenerator',
    'DreamSystem',
    'DreamContent',
    'DreamType',
    'DREAM_GENERATOR_AVAILABLE',
    'DREAM_SYSTEM_AVAILABLE'
]

