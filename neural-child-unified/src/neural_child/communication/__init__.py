#----------------------------------------------------------------------------
#File:       __init__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Package initializer for the communication module.
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Package initializer for the communication module."""

# Import communication components
try:
    from neural_child.communication.message_bus import (
        MessageBus,
        MessageFilter,
        SubscriptionInfo,
        GlobalMessageBus
    )
except ImportError:
    MessageBus = None
    MessageFilter = None
    SubscriptionInfo = None
    GlobalMessageBus = None
    print("Warning: MessageBus components not available.")

__all__ = [
    'MessageBus',
    'MessageFilter',
    'SubscriptionInfo',
    'GlobalMessageBus'
]

