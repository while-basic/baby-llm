"""Language module for neural child development."""

from neural_child.cognitive.language.language_development import (
    LanguageDevelopment,
    LanguageStage,
    WordCategory
)
from neural_child.cognitive.language.text_embed import (
    get_embeddings,
    initialize_embedding_model
)
from neural_child.cognitive.language.symbol_grounding import SymbolGrounding

__all__ = [
    'LanguageDevelopment',
    'LanguageStage',
    'WordCategory',
    'get_embeddings',
    'initialize_embedding_model',
    'SymbolGrounding'
]

