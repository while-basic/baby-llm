#----------------------------------------------------------------------------
#File:       training_system.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Comprehensive training system with monitoring, checkpointing, and early stopping
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Comprehensive training system with monitoring, checkpointing, and early stopping.

Extracted from neural-child-init/training_system.py
Adapted imports to use unified structure.

Note: Some dependencies will be added in later phases:
- curriculum_manager (Phase 2 - development)
- mother_llm (Phase 4 - emotional & interaction)
- metacognition_system (Phase 3 - cognitive systems)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

# Optional imports for dependencies that will be added in later phases
if TYPE_CHECKING:
    from neural_child.development.curriculum_manager import CurriculumManager
    # mother_llm and metacognition_system will be available in later phases

try:
    from neural_child.development.curriculum_manager import CurriculumManager
except ImportError:
    CurriculumManager = None  # Will be available in Phase 2


class MovingAverageMonitor:
    """Monitor training metrics with moving averages."""

    def __init__(self, window_size: int = 50):
        """Initialize moving average monitor.

        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.steps = 0
        self.error_log = []
        self.loss_buffer = deque(maxlen=window_size)
        self.grad_buffer = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.component_losses = defaultdict(lambda: deque(maxlen=window_size))

    def update_stats(
        self,
        total_loss: float,
        individual_losses: dict,
        gradient_norm: float,
        learning_rate: float
    ) -> dict:
        """Update statistics.

        Args:
            total_loss: Total loss value
            individual_losses: Dictionary of individual loss components
            gradient_norm: Gradient norm
            learning_rate: Current learning rate

        Returns:
            Dictionary of statistics
        """
        self.steps += 1
        self.loss_buffer.append(total_loss)
        self.grad_buffer.append(gradient_norm)
        self.learning_rates.append(learning_rate)
        for key, value in individual_losses.items():
            self.component_losses[key].append(value)

        stats = {
            'step': self.steps,
            'total_loss': total_loss,
            'loss_ma': np.mean(self.loss_buffer),
            'loss_std': np.std(self.loss_buffer) if len(self.loss_buffer) > 1 else 0,
            'grad_norm': gradient_norm,
            'grad_ma': np.mean(self.grad_buffer),
            'learning_rate': learning_rate,
            'component_losses': {
                key: {
                    'current': value,
                    'mean': np.mean(list(self.component_losses[key])),
                    'std': (
                        np.std(list(self.component_losses[key]))
                        if len(self.component_losses[key]) > 1 else 0
                    )
                }
                for key, value in individual_losses.items()
            }
        }
        return stats

    def check_loss_spike(self, current_loss: float, threshold: float) -> bool:
        """Check if loss has spiked.

        Args:
            current_loss: Current loss value
            threshold: Threshold for spike detection

        Returns:
            True if loss spike detected
        """
        if len(self.loss_buffer) < 2:
            return False
        loss_mean = np.mean(self.loss_buffer)
        loss_std = np.std(self.loss_buffer)
        if loss_std == 0:
            return current_loss > loss_mean * threshold
        z_score = (current_loss - loss_mean) / loss_std
        return z_score > threshold

    def check_gradient_issues(self, grad_norm: float) -> bool:
        """Check for gradient issues.

        Args:
            grad_norm: Gradient norm

        Returns:
            True if gradient issues detected
        """
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            return True
        if len(self.grad_buffer) < 2:
            return False
        grad_mean = np.mean(self.grad_buffer)
        grad_std = np.std(self.grad_buffer)
        if grad_norm < 1e-7:
            return True
        if grad_std > 0:
            z_score = (grad_norm - grad_mean) / grad_std
            return z_score > 3.0
        return False

    def log_error(self, error_type: str, error_message: str) -> None:
        """Log an error.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.error_log.append({
            'step': self.steps,
            'type': error_type,
            'message': error_message,
            'timestamp': time.time()
        })

    def summarize_episode(self, episode_stats: list) -> dict:
        """Summarize episode statistics.

        Args:
            episode_stats: List of episode statistics

        Returns:
            Episode summary dictionary
        """
        if not episode_stats:
            return {'status': 'failed', 'error_log': self.error_log}

        summary = {
            'steps_completed': len(episode_stats),
            'final_loss': episode_stats[-1]['total_loss'],
            'mean_loss': np.mean([s['total_loss'] for s in episode_stats]),
            'loss_std': np.std([s['total_loss'] for s in episode_stats]),
            'mean_grad_norm': np.mean([s['grad_norm'] for s in episode_stats]),
            'component_trends': {},
            'error_log': self.error_log
        }

        for key in episode_stats[0]['component_losses'].keys():
            values = [
                s['component_losses'][key]['current'] for s in episode_stats
            ]
            summary['component_trends'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'trend': np.polyfit(range(len(values)), values, deg=1)[0]
            }

        return summary


class CheckpointManager:
    """Manage model checkpoints with stability tracking."""

    def __init__(
        self,
        model: nn.Module,
        save_dir: str,
        max_checkpoints: int = 5
    ):
        """Initialize checkpoint manager.

        Args:
            model: Model to checkpoint
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        self.stable_checkpoints = []

    def save_checkpoint(
        self,
        step: int,
        loss: float,
        stats: dict
    ) -> str:
        """Save a checkpoint.

        Args:
            step: Training step
            loss: Current loss
            stats: Training statistics

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        checkpoint_data = {
            'step': step,
            'model_state': self.model.state_dict(),
            'loss': loss,
            'stats': stats,
            'timestamp': time.time()
        }
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'step': step,
            'loss': loss,
            'timestamp': time.time()
        })
        if self._is_stable_checkpoint(stats):
            self.stable_checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()
        return str(checkpoint_path)

    def _is_stable_checkpoint(self, stats: dict) -> bool:
        """Check if checkpoint is stable.

        Args:
            stats: Training statistics

        Returns:
            True if checkpoint is stable
        """
        if 'loss_std' in stats and stats['loss_std'] > 0.5:
            return False
        if ('grad_ma' in stats and
                (stats['grad_ma'] < 1e-7 or stats['grad_ma'] > 10.0)):
            return False
        if 'component_losses' in stats:
            for comp_stats in stats['component_losses'].values():
                if comp_stats['std'] > 0.5:
                    return False
        return True

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        protected = set(self.stable_checkpoints)
        remaining = sorted(
            [cp for cp in self.checkpoint_history if cp['path'] not in protected],
            key=lambda x: x['timestamp']
        )
        while len(remaining) + len(protected) > self.max_checkpoints:
            oldest = remaining.pop(0)
            if oldest['path'].exists():
                oldest['path'].unlink()
            self.checkpoint_history.remove(oldest)

    def load_last_stable(self) -> Optional[dict]:
        """Load last stable checkpoint.

        Returns:
            Checkpoint data or None
        """
        if not self.stable_checkpoints:
            return None
        latest_stable = max(
            self.stable_checkpoints,
            key=lambda p: p.stat().st_mtime
        )
        if latest_stable.exists():
            checkpoint_data = torch.load(latest_stable)
            self.model.load_state_dict(checkpoint_data['model_state'])
            return checkpoint_data
        return None

    def load_best_checkpoint(self, metric: str = 'loss') -> Optional[dict]:
        """Load best checkpoint by metric.

        Args:
            metric: Metric to use for selection

        Returns:
            Checkpoint data or None
        """
        if not self.checkpoint_history:
            return None
        best_checkpoint = min(
            self.checkpoint_history,
            key=lambda x: x[metric]
        )
        if best_checkpoint['path'].exists():
            checkpoint_data = torch.load(best_checkpoint['path'])
            self.model.load_state_dict(checkpoint_data['model_state'])
            return checkpoint_data
        return None


class EarlyStopping:
    """Early stopping based on loss improvement."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """Initialize early stopping.

        Args:
            patience: Number of steps to wait without improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.loss_history = deque(maxlen=100)

    def check(self, current_loss: float) -> bool:
        """Check if training should stop.

        Args:
            current_loss: Current loss value

        Returns:
            True if training should stop
        """
        self.loss_history.append(current_loss)
        if len(self.loss_history) < self.patience:
            return False
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        if len(self.loss_history) >= 10:
            recent_mean = np.mean(list(self.loss_history)[-10:])
            recent_std = np.std(list(self.loss_history)[-10:])
            if recent_std > recent_mean * 0.5:
                return True
        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.loss_history.clear()

