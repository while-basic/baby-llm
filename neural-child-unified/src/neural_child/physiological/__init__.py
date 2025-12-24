"""Physiological systems for neural child development."""

# Import physiological components
try:
    from neural_child.physiological.heartbeat_system import (
        HeartbeatSystem,
        HeartRateState
    )
except ImportError:
    HeartbeatSystem = None
    HeartRateState = None
    print("Warning: HeartbeatSystem not available.")

__all__ = [
    'HeartbeatSystem',
    'HeartRateState'
]

