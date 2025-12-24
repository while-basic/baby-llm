"""LLM integration module for neural child development."""

# Import LLM module functions
try:
    from neural_child.interaction.llm.llm_module import chat_completion
except ImportError:
    chat_completion = None
    print("Warning: chat_completion not available.")

# Import Ollama chat classes
try:
    from neural_child.interaction.llm.ollama_chat import (
        OllamaChat,
        OllamaChildChat,
        get_child_response,
        analyze_sentiment
    )
except ImportError:
    OllamaChat = None
    OllamaChildChat = None
    get_child_response = None
    analyze_sentiment = None
    print("Warning: OllamaChat classes not available.")

# Import Mother LLM
try:
    from neural_child.interaction.llm.mother_llm import (
        MotherLLM,
        MotherResponse
    )
except ImportError:
    MotherLLM = None
    MotherResponse = None
    print("Warning: MotherLLM not available.")

__all__ = [
    'chat_completion',
    'OllamaChat',
    'OllamaChildChat',
    'get_child_response',
    'analyze_sentiment',
    'MotherLLM',
    'MotherResponse'
]

