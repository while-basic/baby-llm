#----------------------------------------------------------------------------
#File:       __init__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Package initializer for the safety module.
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Package initializer for the safety module."""

# Import safety components
try:
    from neural_child.safety.safety_monitor import (
        SafetyMonitor,
        SafetyException,
        InteractionSafety  # Alias for backward compatibility
    )
except ImportError:
    SafetyMonitor = None
    SafetyException = None
    InteractionSafety = None
    print("Warning: SafetyMonitor components not available.")

__all__ = [
    'SafetyMonitor',
    'SafetyException',
    'InteractionSafety'
]

