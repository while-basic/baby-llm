"""Training module for neural child development."""

from neural_child.core.training.self_supervised_trainer import SelfSupervisedTrainer
from neural_child.core.training.training_system import (
    MovingAverageMonitor,
    CheckpointManager,
    EarlyStopping
)
from neural_child.core.training.replay_system import ReplayOptimizer

__all__ = [
    'SelfSupervisedTrainer',
    'MovingAverageMonitor',
    'CheckpointManager',
    'EarlyStopping',
    'ReplayOptimizer'
]

