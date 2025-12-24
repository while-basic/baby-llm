"""Chat systems for neural child development."""

# Import chat systems (with optional dependencies)
try:
    from neural_child.interaction.chat.integrated_chat import IntegratedChatSystem
except ImportError:
    IntegratedChatSystem = None
    print("Warning: IntegratedChatSystem not available.")

try:
    from neural_child.interaction.chat.emotional_chat import EmotionalChatSystem
except ImportError:
    EmotionalChatSystem = None
    print("Warning: EmotionalChatSystem not available.")

try:
    from neural_child.interaction.chat.self_awareness_chat import SelfAwarenessChatInterface
except ImportError:
    SelfAwarenessChatInterface = None
    print("Warning: SelfAwarenessChatInterface not available.")

__all__ = [
    'IntegratedChatSystem',
    'EmotionalChatSystem',
    'SelfAwarenessChatInterface'
]

