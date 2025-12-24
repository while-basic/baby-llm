#----------------------------------------------------------------------------
#File:       symbol_grounding.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Symbol grounding for neural child development
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Symbol grounding for neural child development.

Extracted from neural-child-init/symbol_grounding.py
Adapted imports to use unified structure.
"""

import torch
from typing import Dict, List, Tuple, Optional

# Import from unified structure
from neural_child.cognitive.language.text_embed import get_embeddings


class SymbolGrounding:
    """Symbol grounding system for mapping concepts to tokens."""

    def __init__(self, device: str = 'cuda'):
        """Initialize symbol grounding system.

        Args:
            device: Device to use for tensors
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.concept_map: Dict[str, torch.Tensor] = {}
        self.reverse_map: Dict[Tuple, str] = {}
        self.embedding_matrix = torch.empty((0, 768), device=self.device)

    def add_symbol(self, concept: str, token: str) -> None:
        """Add a symbol mapping from concept to token.

        Args:
            concept: Concept string
            token: Token string
        """
        embedding_data = get_embeddings(concept)
        if embedding_data and 'data' in embedding_data:
            embedding = torch.tensor(
                embedding_data['data'][0]['embedding'],
                device=self.device
            )
            self.concept_map[token] = embedding
            self.reverse_map[tuple(embedding.cpu().numpy())] = token
            self.embedding_matrix = torch.cat(
                [self.embedding_matrix, embedding.unsqueeze(0)], dim=0
            )

    def get_token(self, embedding: torch.Tensor) -> Optional[str]:
        """Get token for given embedding.

        Args:
            embedding: Embedding tensor

        Returns:
            Token string or None
        """
        if self.embedding_matrix.size(0) == 0:
            return None

        embedding = embedding.to(self.device)
        similarities = torch.matmul(self.embedding_matrix, embedding)
        closest_idx = torch.argmax(similarities)
        closest_embedding = self.embedding_matrix[closest_idx]
        embedding_tuple = tuple(closest_embedding.cpu().numpy())
        return self.reverse_map.get(embedding_tuple)

    def batch_ground(self, concepts: List[str]) -> Dict[str, Dict]:
        """Ground multiple concepts to tokens.

        Args:
            concepts: List of concept strings

        Returns:
            Dictionary mapping concepts to token and embedding info
        """
        embeddings_data = get_embeddings(concepts)
        result = {}
        if embeddings_data and 'data' in embeddings_data:
            for i, concept in enumerate(concepts):
                if i < len(embeddings_data['data']):
                    emb = embeddings_data['data'][i]
                    result[concept] = {
                        'token': f"[{concept.upper()}]",
                        'embedding': torch.tensor(
                            emb['embedding'],
                            device=self.device
                        )
                    }
        return result

