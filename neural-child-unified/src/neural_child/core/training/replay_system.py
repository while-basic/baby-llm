#----------------------------------------------------------------------------
#File:       replay_system.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Replay system for experience replay and importance sampling
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Replay system for experience replay and importance sampling.

Extracted from neural-child-init/replay_system.py
Adapted imports to use unified structure.
"""

import random
import torch
from torch import nn
from typing import List, Tuple, Any, Optional


class ReplayOptimizer:
    """Replay optimizer with importance sampling and memory pruning."""

    def __init__(self, memory_capacity: int = 10000, device: str = 'cuda'):
        """Initialize replay optimizer.

        Args:
            memory_capacity: Maximum number of experiences to store
            device: Device to use for tensors
        """
        self.memory: List[Any] = []
        self.capacity = memory_capacity
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.importance_weights = nn.Parameter(
            torch.ones(memory_capacity, device=self.device)
        )
        self.decay_factor = 0.99

    def add_experience(self, experience: Any) -> None:
        """Add experience to replay memory.

        Args:
            experience: Experience to add
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.memory[idx] = experience

    def sample_batch(
        self,
        batch_size: int = 32
    ) -> Tuple[List[Any], List[int]]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (samples, indices)
        """
        # Use the available number of experiences if fewer than requested
        actual_batch_size = min(batch_size, len(self.memory))
        if actual_batch_size == 0:
            return [], []

        indices = random.sample(range(len(self.memory)), actual_batch_size)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_weights(
        self,
        indices: List[int],
        losses: torch.Tensor
    ) -> None:
        """Update importance weights based on losses.

        Args:
            indices: Indices of experiences used
            losses: Loss values for each experience
        """
        self.importance_weights.data *= self.decay_factor
        for i, loss in zip(indices, losses):
            if i < len(self.importance_weights):
                self.importance_weights.data[i] += loss.item()

        # Prune memory if near capacity
        if len(self.memory) > self.capacity * 0.95:
            prune_idx = torch.argsort(self.importance_weights)[
                :len(self.memory) // 20
            ]
            self.memory = [
                m for i, m in enumerate(self.memory)
                if i not in prune_idx.tolist()
            ]
            self.importance_weights = nn.Parameter(
                torch.ones(len(self.memory), device=self.device),
                requires_grad=True
            )

