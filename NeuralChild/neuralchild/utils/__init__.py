# Copyright (c) 2025 Celaya Solutions AI Research Lab

"""Utility modules for NeuralChild project."""

from neuralchild.utils.llm_module import (
    chat_completion,
    get_embeddings,
    simulate_llm_response,
    LLMError
)

__all__ = [
    'chat_completion',
    'get_embeddings',
    'simulate_llm_response',
    'LLMError'
]
