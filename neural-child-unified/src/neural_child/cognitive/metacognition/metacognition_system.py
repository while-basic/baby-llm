#----------------------------------------------------------------------------
#File:       metacognition_system.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Unified metacognition and self-awareness system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Unified metacognition and self-awareness system for neural child development.

Merged from:
- neural-child-init/metacognition.py (hypothesis networks, self-correction)
- neural-child-init/self_awareness_network.py (self-awareness levels, emotional memory)

Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import json

# Optional import for networkx (used for self-concept graph)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class SelfAwarenessLevel(Enum):
    """Levels of self-awareness development."""
    PHYSICAL = auto()      # Basic physical self-awareness (0-18 months)
    MIRROR = auto()        # Mirror self-recognition (18-24 months)
    EMOTIONAL = auto()     # Emotional self-awareness (2-3 years)
    COGNITIVE = auto()     # Understanding own thoughts (3-4 years)
    METACOGNITIVE = auto() # Thinking about thinking (4-5 years)
    SOCIAL = auto()        # Social self-awareness (5+ years)
    ABSTRACT = auto()      # Abstract self-concept (adolescence)


class MetacognitionSystem(nn.Module):
    """Unified metacognition and self-awareness system."""

    def __init__(self, base_dim: int = 128, num_hypotheses: int = 5, device: str = 'cpu'):
        """Initialize metacognition system.

        Args:
            base_dim: Base dimension for embeddings
            num_hypotheses: Number of hypothesis networks for self-correction
            device: Device to use for computation
        """
        super().__init__()
        self.base_dim = base_dim
        self.num_hypotheses = num_hypotheses
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = base_dim

        # Feature dimensions for self-awareness
        self.physical_dim = base_dim // 4
        self.emotional_dim = base_dim // 4
        self.cognitive_dim = base_dim // 4
        self.attention_dim = base_dim // 4

        # Base metacognition network (from metacognition.py)
        self.base_network = nn.Sequential(
            nn.Linear(base_dim, 256),
            nn.GELU(),
            nn.Linear(256, base_dim)
        )

        # Hypothesis networks for self-correction
        self.hypothesis_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, 256),
                nn.GELU(),
                nn.Linear(256, base_dim)
            ) for _ in range(num_hypotheses)
        ])

        # Critic network for evaluating hypotheses
        self.critic = nn.Sequential(
            nn.Linear(base_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Bayesian layer for uncertainty estimation
        self.bayesian_layer = nn.LSTM(base_dim, base_dim)

        # Complexity head
        self.complexity_head = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Linear(base_dim // 2, 1),
            nn.Sigmoid()
        )

        # Self-awareness networks (from self_awareness_network.py)
        self.input_projection = nn.Linear(384, base_dim)

        # Feature networks
        self.physical_network = nn.Sequential(
            nn.Linear(base_dim, self.physical_dim),
            nn.ReLU(),
            nn.Linear(self.physical_dim, self.physical_dim)
        )

        self.emotional_network = nn.Sequential(
            nn.Linear(base_dim, self.emotional_dim),
            nn.ReLU(),
            nn.Linear(self.emotional_dim, self.emotional_dim)
        )

        self.cognitive_network = nn.Sequential(
            nn.Linear(base_dim, self.cognitive_dim),
            nn.ReLU(),
            nn.Linear(self.cognitive_dim, self.cognitive_dim)
        )

        self.attention_network = nn.Sequential(
            nn.Linear(base_dim, self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, self.attention_dim)
        )

        # Memory compression
        self.memory_compression = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, base_dim // 2)
        )

        # Emotional memory network
        self.emotional_memory = nn.LSTM(
            input_size=base_dim // 2 + 4,  # Compressed features + emotions
            hidden_size=base_dim,
            num_layers=2,
            batch_first=True
        )

        # Word learning network
        self.word_learning = nn.Sequential(
            nn.Linear(base_dim + 384, base_dim),  # Word embedding dim = 384
            nn.ReLU(),
            nn.Linear(base_dim, 1),
            nn.Sigmoid()
        )

        # Emotional association network
        self.emotional_association = nn.Sequential(
            nn.Linear(base_dim + 384, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, 4),  # 4 emotion dimensions
            nn.Sigmoid()
        )

        # Initialize states (register as buffers)
        self.register_buffer('emotional_state', torch.zeros(1, 4))  # [joy, trust, fear, surprise]
        self.register_buffer('attention_state', torch.zeros(1, self.attention_dim))
        self.metacognition = 0.0

        # Storage
        self.emotional_memories: List[Dict] = []
        self.known_words = set()
        self.word_emotions: Dict[str, torch.Tensor] = {}

        # Self-concept graph (optional)
        if NETWORKX_AVAILABLE:
            self.self_concept_graph = nx.Graph()
        else:
            self.self_concept_graph = None
            print("Warning: networkx not available. Self-concept graph disabled.")

        # Development tracking
        self.current_level = SelfAwarenessLevel.PHYSICAL
        self.level_progress = 0.0
        self.reflection_history: List[Dict] = []

        # Move to device
        self.to(self.device)

    def forward(self, input_embedding: torch.Tensor) -> Dict[str, Any]:
        """Process input through metacognition system.

        Args:
            input_embedding: Input embedding tensor

        Returns:
            Dictionary with thought, confidence, uncertainty, complexity, and features
        """
        # Base metacognition processing
        base_output = self.base_network(input_embedding)
        combined = torch.cat([input_embedding, base_output], dim=-1)
        confidence = self.critic(combined)

        # Bayesian uncertainty estimation
        _, (hidden, _) = self.bayesian_layer(base_output.unsqueeze(0))
        uncertainty = hidden.squeeze(0)

        # Complexity estimation
        complexity = self.complexity_head(base_output)

        # Self-awareness processing (if input is 384-dim)
        if input_embedding.size(-1) == 384:
            x = self.input_projection(input_embedding)
            batch_size = x.size(0)

            # Process features
            physical_features = self.physical_network(x)
            emotional_features = self.emotional_network(x)
            cognitive_features = self.cognitive_network(x)
            attention_features = self.attention_network(x)

            # Combine features
            combined_features = torch.cat([
                physical_features,
                emotional_features,
                cognitive_features,
                attention_features
            ], dim=1)

            # Compress features for memory
            compressed_features = self.memory_compression(x)

            # Update emotional memory
            if len(self.emotional_memories) > 0:
                memory_input = torch.cat([
                    compressed_features,
                    self.emotional_state.expand(batch_size, -1)
                ], dim=1).unsqueeze(1)

                memory_output, _ = self.emotional_memory(memory_input)
                memory_output = memory_output.squeeze(1)
            else:
                memory_output = torch.zeros(batch_size, self.hidden_dim, device=x.device)

            # Calculate metacognition
            self.metacognition = torch.mean(attention_features).item()

            # Update emotional state
            emotional_update = torch.sigmoid(memory_output[:, :4]).detach()
            new_state = (0.9 * self.emotional_state + 0.1 * emotional_update).detach()
            self.emotional_state.data.copy_(new_state)

            return {
                'thought': base_output,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'complexity': complexity,
                'physical_features': physical_features,
                'emotional_features': emotional_features,
                'cognitive_features': cognitive_features,
                'attention_features': attention_features,
                'combined_features': combined_features,
                'compressed_features': compressed_features,
                'memory_output': memory_output,
                'metacognition': self.metacognition
            }
        else:
            return {
                'thought': base_output,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'complexity': complexity
            }

    def self_correct(self, thought_embedding: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Self-correct thought by generating and evaluating alternatives.

        Args:
            thought_embedding: Current thought embedding
            temperature: Temperature for noise generation

        Returns:
            Corrected thought embedding
        """
        alternatives = []
        scores = []
        base_result = self.forward(thought_embedding)
        base_uncertainty = base_result['uncertainty']

        for i, net in enumerate(self.hypothesis_network):
            noise = torch.randn_like(thought_embedding) * temperature * (i + 1) / self.num_hypotheses
            alt = net(thought_embedding + noise)
            alternatives.append(alt)
            combined = torch.cat([thought_embedding, alt], -1)
            score = self.critic(combined)
            scores.append(score)

        if not alternatives:
            return thought_embedding

        weighted_scores = torch.stack(scores) * (1 - base_uncertainty)
        best_idx = torch.argmax(weighted_scores)
        return alternatives[best_idx]

    def update_self_concept(self, interaction: Dict):
        """Update self-concept graph with new interaction.

        Args:
            interaction: Dictionary with 'content' and 'emotional_state'
        """
        if not NETWORKX_AVAILABLE or self.self_concept_graph is None:
            return

        # Add new node for current interaction
        node_id = len(self.self_concept_graph.nodes)
        self.self_concept_graph.add_node(
            node_id,
            content=interaction['content'],
            emotions=interaction['emotional_state']
        )

        # Connect to similar nodes
        for other_id, other_data in self.self_concept_graph.nodes(data=True):
            if other_id != node_id:
                # Calculate emotional similarity
                current_emotions = torch.tensor([
                    interaction['emotional_state']['joy'],
                    interaction['emotional_state']['trust'],
                    interaction['emotional_state']['fear'],
                    interaction['emotional_state']['surprise']
                ])

                other_emotions = torch.tensor([
                    other_data['emotions']['joy'],
                    other_data['emotions']['trust'],
                    other_data['emotions']['fear'],
                    other_data['emotions']['surprise']
                ])

                similarity = F.cosine_similarity(
                    current_emotions.unsqueeze(0),
                    other_emotions.unsqueeze(0)
                ).item()

                # Add edge if similarity is high enough
                if similarity > 0.3:
                    self.self_concept_graph.add_edge(
                        node_id,
                        other_id,
                        weight=similarity
                    )

    def get_development_metrics(self) -> Dict:
        """Get current development metrics.

        Returns:
            Dictionary with development metrics
        """
        metrics = {
            'known_words': len(self.known_words),
            'emotional_memories': len(self.emotional_memories),
            'metacognition': self.metacognition,
            'self_awareness_level': self.current_level.name,
            'level_progress': self.level_progress
        }

        if NETWORKX_AVAILABLE and self.self_concept_graph is not None:
            metrics['self_concept_size'] = len(self.self_concept_graph.nodes())
            metrics['self_concept_connections'] = len(self.self_concept_graph.edges())
        else:
            metrics['self_concept_size'] = 0
            metrics['self_concept_connections'] = 0

        return metrics

    def learn_word(self, word: str, embedding: torch.Tensor, features: torch.Tensor) -> Dict:
        """Learn a new word and its emotional associations.

        Args:
            word: Word to learn
            embedding: Word embedding tensor
            features: Context features tensor

        Returns:
            Dictionary with learning results
        """
        # Ensure embeddings have batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Combine features with word embedding
        combined = torch.cat([features, embedding], dim=1)

        # Calculate learning confidence
        learning_confidence = self.word_learning(combined)

        # Get emotional association if confidence is high enough
        result = {
            'word': word,
            'learning_confidence': float(learning_confidence.squeeze().item())
        }

        if result['learning_confidence'] > 0.5:
            emotional_assoc = self.emotional_association(combined)
            self.known_words.add(word)
            self.word_emotions[word] = emotional_assoc.detach()
            result['emotional_association'] = emotional_assoc.squeeze().tolist()

        return result

    def get_word_emotion(self, word: str) -> Optional[Dict[str, Any]]:
        """Get emotional association for a word.

        Args:
            word: Word to look up

        Returns:
            Dictionary with word and emotions, or None if not found
        """
        word = word.lower()
        if word in self.word_emotions:
            emotion_data = self.word_emotions[word]
            return {
                'word': word,
                'emotions': emotion_data.tolist()
            }
        return None

    def save_state(self, path: str):
        """Save network state.

        Args:
            path: Path to save state file
        """
        state = {
            'emotional_state': self.emotional_state.tolist(),
            'attention_state': self.attention_state.tolist(),
            'metacognition': self.metacognition,
            'known_words': list(self.known_words),
            'word_emotions': {
                word: tensor.tolist()
                for word, tensor in self.word_emotions.items()
            },
            'emotional_memories': self.emotional_memories,
            'current_level': self.current_level.name,
            'level_progress': self.level_progress
        }

        if NETWORKX_AVAILABLE and self.self_concept_graph is not None:
            state['self_concept_graph'] = {
                'nodes': list(self.self_concept_graph.nodes(data=True)),
                'edges': list(self.self_concept_graph.edges(data=True))
            }

        torch.save(state, path)

    def load_state(self, path: str):
        """Load network state.

        Args:
            path: Path to load state file from
        """
        state = torch.load(path, map_location=self.device)

        # Restore buffers
        self.emotional_state.data.copy_(torch.tensor(state['emotional_state'], device=self.device))
        self.attention_state.data.copy_(torch.tensor(state['attention_state'], device=self.device))
        self.metacognition = state['metacognition']
        self.known_words = set(state['known_words'])
        self.word_emotions = {
            word: torch.tensor(data, device=self.device)
            for word, data in state['word_emotions'].items()
        }
        self.emotional_memories = state['emotional_memories']
        self.current_level = SelfAwarenessLevel[state.get('current_level', 'PHYSICAL')]
        self.level_progress = state.get('level_progress', 0.0)

        # Rebuild graph if available
        if NETWORKX_AVAILABLE and 'self_concept_graph' in state:
            self.self_concept_graph = nx.Graph()
            for node, data in state['self_concept_graph']['nodes']:
                self.self_concept_graph.add_node(node, **data)
            for u, v, data in state['self_concept_graph']['edges']:
                self.self_concept_graph.add_edge(u, v, **data)

