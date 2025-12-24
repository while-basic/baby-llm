"""Psychological components for neural child development."""

# Import psychological components
try:
    from neural_child.psychological.attachment import AttachmentSystem
except ImportError:
    AttachmentSystem = None
    print("Warning: AttachmentSystem not available.")

try:
    from neural_child.psychological.theory_of_mind import TheoryOfMind
except ImportError:
    TheoryOfMind = None
    print("Warning: TheoryOfMind not available.")

try:
    from neural_child.psychological.defense_mechanisms import DefenseMechanisms
except ImportError:
    DefenseMechanisms = None
    print("Warning: DefenseMechanisms not available.")

__all__ = [
    'AttachmentSystem',
    'TheoryOfMind',
    'DefenseMechanisms'
]

