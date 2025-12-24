"""Emotional module for neural child development."""

# Import regulation
try:
    from neural_child.emotional.regulation import (
        EmotionalRegulation,
        EmotionalState
    )
except ImportError:
    EmotionalRegulation = None
    EmotionalState = None

# Import development
try:
    from neural_child.emotional.development import (
        EmotionalDevelopmentSystem,
        EmotionalCapability,
        EmotionalState as DevelopmentEmotionalState
    )
except ImportError:
    EmotionalDevelopmentSystem = None
    EmotionalCapability = None
    DevelopmentEmotionalState = None

# Import memory
try:
    from neural_child.emotional.memory import (
        EmotionalMemorySystem,
        EmotionalMemoryEntry,
        EmotionalAssociation,
        EmotionalMemoryProcessor
    )
except ImportError:
    EmotionalMemorySystem = None
    EmotionalMemoryEntry = None
    EmotionalAssociation = None
    EmotionalMemoryProcessor = None

# Import embedding
try:
    from neural_child.emotional.embedding import (
        EmotionalEmbedder,
        QuantumEmotionalProcessor
    )
except ImportError:
    EmotionalEmbedder = None
    QuantumEmotionalProcessor = None

__all__ = [
    'EmotionalRegulation',
    'EmotionalState',
    'EmotionalDevelopmentSystem',
    'EmotionalCapability',
    'DevelopmentEmotionalState',
    'EmotionalMemorySystem',
    'EmotionalMemoryEntry',
    'EmotionalAssociation',
    'EmotionalMemoryProcessor',
    'EmotionalEmbedder',
    'QuantumEmotionalProcessor'
]

