#----------------------------------------------------------------------------
#File:       text_embed.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Text embedding module for neural child development
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Text embedding module for neural child development.

Extracted from neural-child-init/text_embed.py
Adapted imports to use unified structure.
"""

import torch
import numpy as np

# Optional imports for transformers
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

# Initialize the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None


def initialize_embedding_model() -> None:
    """Initialize the embedding model and tokenizer."""
    global tokenizer, model
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library not available")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()


def get_embeddings(text: str, normalize: bool = True) -> dict:
    """Get embeddings for input text.

    Args:
        text: Input text to embed
        normalize: Whether to normalize embeddings

    Returns:
        Dictionary with embedding data
    """
    try:
        # Initialize if not already done
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        if tokenizer is None or model is None:
            initialize_embedding_model()

        # Tokenize and prepare input
        inputs = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = (
            torch.sum(token_embeddings * input_mask_expanded, 1) /
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and return in expected format
        embedding_np = embeddings.cpu().numpy()
        return {
            'data': [{
                'embedding': embedding_np[0].tolist()
            }]
        }

    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return zero embedding as fallback
        return {
            'data': [{
                'embedding': np.zeros(384).tolist()  # Default embedding size
            }]
        }

