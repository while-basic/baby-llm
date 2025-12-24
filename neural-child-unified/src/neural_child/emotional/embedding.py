#----------------------------------------------------------------------------
#File:       embedding.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Emotional embedding with quantum-inspired processing
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Emotional embedding system with quantum-inspired processing for neural child development.

Merged from:
- neural-child-init/emotional_embedding.py (standard embedding)
- neural-child-2/main.py (quantum emotional processing)

Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from collections import deque

# Optional import for text embeddings
try:
    from neural_child.cognitive.language.text_embed import get_embeddings
    TEXT_EMBED_AVAILABLE = True
except ImportError:
    TEXT_EMBED_AVAILABLE = False
    get_embeddings = None


class EmotionalEmbedder(nn.Module):
    """Standard emotional embedder for text-to-emotion mapping."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.valence_proj = nn.Linear(768, 4)  # Projects to 4 emotion dimensions
        self.valence_proj.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, text_input):
        """Embed text and project to emotional dimensions."""
        if not TEXT_EMBED_AVAILABLE or get_embeddings is None:
            # Fallback: return zero embeddings
            embeddings = torch.zeros(1, 768, device=self.device, dtype=torch.float32)
        else:
            embed_result = get_embeddings(text_input)
            embeddings = torch.tensor(
                [item['embedding'] for item in embed_result], 
                device=self.device,
                dtype=torch.float32
            )
        
        valence_arousal = torch.sigmoid(self.valence_proj(embeddings))
        return {
            'semantic_embedding': embeddings,
            'valence': valence_arousal[:, 0],
            'arousal': valence_arousal[:, 1]
        }


class QuantumEmotionalProcessor:
    """Quantum-inspired emotional processing system.
    
    Features:
    - 8-dimensional emotional superposition
    - Emotional entanglement matrix
    - Coherence tracking and decoherence
    - State collapse mechanics
    """
    
    def __init__(self, device='cuda', emotion_dim=8):
        """Initialize quantum emotional processor."""
        self.device = device
        self.emotion_dim = emotion_dim
        
        # Quantum emotional state
        self.quantum_emotional_state = {
            'superposition': torch.zeros(emotion_dim, device=device),  # Extended emotional dimensions
            'entanglement_matrix': torch.eye(emotion_dim, device=device),  # Emotional entanglement
            'coherence_factor': 1.0,  # Quantum coherence of emotional states
            'collapse_threshold': 0.7  # Threshold for emotional state collapse
        }
        
        # Track quantum metrics
        self.quantum_metrics = {
            'coherence_history': deque(maxlen=1000),
            'entanglement_strength': deque(maxlen=1000),
            'superposition_stability': deque(maxlen=1000)
        }
    
    def process_quantum_emotions(self, stimulus_vector: torch.Tensor) -> torch.Tensor:
        """Process emotions using quantum-inspired algorithms.
        
        Args:
            stimulus_vector: Input stimulus vector
            
        Returns:
            Processed quantum emotional state
        """
        # Ensure stimulus vector is properly shaped
        if stimulus_vector.dim() == 0:
            stimulus_vector = stimulus_vector.unsqueeze(0)  # Make it 1D
        if stimulus_vector.dim() == 1:
            stimulus_vector = stimulus_vector.unsqueeze(0)  # Add batch dimension
            
        # Reshape entanglement matrix if needed
        entanglement_matrix = self.quantum_emotional_state['entanglement_matrix']
        if entanglement_matrix.dim() == 2:
            entanglement_matrix = entanglement_matrix.unsqueeze(0)
        else:
            entanglement_matrix = entanglement_matrix
            
        # Ensure dimensions match
        if stimulus_vector.size(-1) != entanglement_matrix.size(-1):
            # Project stimulus to match entanglement matrix dimensions
            if stimulus_vector.size(-1) < entanglement_matrix.size(-1):
                padding = torch.zeros(
                    stimulus_vector.size(0),
                    entanglement_matrix.size(-1) - stimulus_vector.size(-1),
                    device=self.device
                )
                stimulus_vector = torch.cat([stimulus_vector, padding], dim=-1)
            else:
                stimulus_vector = stimulus_vector[:, :entanglement_matrix.size(-1)]
        
        # Update superposition state with proper broadcasting
        self.quantum_emotional_state['superposition'] = F.softmax(
            torch.matmul(stimulus_vector, entanglement_matrix).squeeze(), 
            dim=-1
        )
        
        # Apply quantum noise (decoherence)
        noise = torch.randn_like(self.quantum_emotional_state['superposition']) * (
            1 - self.quantum_emotional_state['coherence_factor']
        )
        self.quantum_emotional_state['superposition'] += noise
        
        # Update coherence
        self.quantum_emotional_state['coherence_factor'] *= 0.99  # Gradual decoherence
        
        # Check for emotional collapse
        if torch.max(self.quantum_emotional_state['superposition']) > self.quantum_emotional_state['collapse_threshold']:
            # Collapse to classical emotional state
            classical_state = torch.zeros_like(self.quantum_emotional_state['superposition'])
            max_idx = torch.argmax(self.quantum_emotional_state['superposition'])
            classical_state[max_idx] = 1.0
            self.quantum_emotional_state['superposition'] = classical_state
            
        # Update metrics
        self.quantum_metrics['coherence_history'].append(
            self.quantum_emotional_state['coherence_factor']
        )
        self.quantum_metrics['entanglement_strength'].append(
            torch.trace(self.quantum_emotional_state['entanglement_matrix']).item()
        )
        self.quantum_metrics['superposition_stability'].append(
            torch.std(self.quantum_emotional_state['superposition']).item()
        )
        
        return self.quantum_emotional_state['superposition']
    
    def get_quantum_state(self) -> Dict[str, Any]:
        """Get current quantum emotional state."""
        return {
            'superposition': self.quantum_emotional_state['superposition'].clone(),
            'entanglement_matrix': self.quantum_emotional_state['entanglement_matrix'].clone(),
            'coherence_factor': self.quantum_emotional_state['coherence_factor'],
            'collapse_threshold': self.quantum_emotional_state['collapse_threshold']
        }
    
    def get_quantum_metrics(self) -> Dict[str, List[float]]:
        """Get quantum processing metrics."""
        return {
            'coherence_history': list(self.quantum_metrics['coherence_history']),
            'entanglement_strength': list(self.quantum_metrics['entanglement_strength']),
            'superposition_stability': list(self.quantum_metrics['superposition_stability'])
        }
    
    def reset_quantum_state(self):
        """Reset quantum emotional state to initial conditions."""
        self.quantum_emotional_state = {
            'superposition': torch.zeros(self.emotion_dim, device=self.device),
            'entanglement_matrix': torch.eye(self.emotion_dim, device=self.device),
            'coherence_factor': 1.0,
            'collapse_threshold': 0.7
        }
        self.quantum_metrics = {
            'coherence_history': deque(maxlen=1000),
            'entanglement_strength': deque(maxlen=1000),
            'superposition_stability': deque(maxlen=1000)
        }

