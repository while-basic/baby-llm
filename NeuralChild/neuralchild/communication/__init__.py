"""Communication module for the NeuralChild project.

Copyright (c) 2025 Celaya Solutions AI Research Lab

This module provides message bus and communication infrastructure
for inter-network communication in the artificial mind.
"""

from neuralchild.communication.message_bus import MessageBus, MessageFilter, GlobalMessageBus

__all__ = ['MessageBus', 'MessageFilter', 'GlobalMessageBus']
