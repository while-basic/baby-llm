#----------------------------------------------------------------------------
#File:       autonomous_learner.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Autonomous learning system for curiosity-driven self-directed learning
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Autonomous learning system for curiosity-driven self-directed learning.

Extracted from neural-child-1/autonomous_learner.py and autonomous_learning.py
Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np

# Optional imports for unified structure
try:
    from neural_child.config.config import Config
    config = Config()
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = None
    print("Warning: Config not available. Using default values.")


class AutonomousLearner:
    """Autonomous learning system for curiosity-driven self-directed learning.
    
    This system enables the neural child to learn independently by:
    - Generating self-directed learning tasks
    - Self-evaluating performance
    - Adapting learning parameters based on performance
    - Adjusting curriculum difficulty dynamically
    """
    
    def __init__(self, child_model: Optional[Any] = None, device: Optional[torch.device] = None):
        """Initialize the autonomous learner.
        
        Args:
            child_model: Reference to the child model/brain (optional)
            device: Device to run on (defaults to cuda if available)
        """
        self.child = child_model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Learning parameters
        default_learning_rate = config.learning_rate if CONFIG_AVAILABLE and hasattr(config, 'learning_rate') else 1e-4
        self.learning_parameters = {
            'learning_rate': default_learning_rate,
            'exploration_rate': 0.3,
            'curriculum_difficulty': 0.1
        }
        self.performance_history: List[float] = []
        self.exploration_decay = 0.995
        self.min_exploration = 0.05
        self.curiosity_threshold = 0.7
        
    def get_learning_rate(self) -> float:
        """Get the current learning rate.
        
        Returns:
            The current learning rate value.
        """
        return self.learning_parameters['learning_rate']

    def learn_independently(self) -> Dict[str, Any]:
        """Execute one cycle of autonomous learning.
        
        Returns:
            Dictionary with performance metrics and learning parameters
        """
        try:
            # Generate self-directed learning task
            task = self._generate_task()
            
            # Attempt the task
            with torch.no_grad():
                if self.child and hasattr(self.child, 'brain'):
                    response = self.child.brain(task['input'])
                elif self.child and hasattr(self.child, 'forward'):
                    response = self.child(task['input'])
                else:
                    # Fallback: generate dummy response
                    response = torch.randn_like(task['target'])
            
            # Self-evaluate performance
            performance = self._evaluate_performance(response, task['target'])
            
            # Update learning parameters based on performance
            self._adapt_parameters(performance)
            
            # Record performance
            self.performance_history.append(performance)
            
            # Decay exploration rate
            self.learning_parameters['exploration_rate'] = max(
                self.min_exploration,
                self.learning_parameters['exploration_rate'] * self.exploration_decay
            )
            
            return {
                'performance': performance,
                'learning_rate': self.learning_parameters['learning_rate'],
                'exploration_rate': self.learning_parameters['exploration_rate'],
                'curriculum_difficulty': self.learning_parameters['curriculum_difficulty'],
                'task_difficulty': task.get('difficulty', 0.1)
            }
            
        except Exception as e:
            print(f"Error in autonomous learning: {str(e)}")
            return {
                'performance': 0.0,
                'learning_rate': self.learning_parameters['learning_rate'],
                'exploration_rate': self.learning_parameters['exploration_rate'],
                'curriculum_difficulty': self.learning_parameters['curriculum_difficulty'],
                'error': str(e)
            }
    
    def generate_self_prompt(self) -> str:
        """Generate learning prompts based on curiosity.
        
        Returns:
            A self-generated learning prompt
        """
        topics = [
            "feelings", "objects", "people", "actions",
            "colors", "numbers", "words", "concepts"
        ]
        
        # Select topic based on current development stage
        stage_value = 0.5  # Default
        if self.child and hasattr(self.child, 'curriculum'):
            if hasattr(self.child.curriculum, 'current_stage'):
                stage = self.child.curriculum.current_stage
                if hasattr(stage, 'value'):
                    stage_value = stage.value
                elif isinstance(stage, int):
                    stage_value = float(stage)
        
        complexity = min(1.0, stage_value / 17.0)  # Normalize stage value
        
        # Select random topic
        topic = np.random.choice(topics)
        
        return f"I want to learn about {topic} at complexity level {complexity:.2f}"
    
    def evaluate_learning(self, response: Dict[str, Any]) -> float:
        """Self-evaluate learning progress.
        
        Args:
            response: Response dictionary with confidence, coherence, novelty
            
        Returns:
            Learning evaluation score (0.0 to 1.0)
        """
        confidence = response.get('confidence', 0.0)
        coherence = response.get('coherence', 0.0)
        novelty = response.get('novelty', 0.0)
        
        return (confidence + coherence + novelty) / 3.0

    def adjust_learning_path(self, performance: float):
        """Adjust learning parameters based on performance.
        
        Args:
            performance: Performance score (0.0 to 1.0)
        """
        if performance < 0.3:
            self.learning_parameters['exploration_rate'] *= 0.9  # Reduce exploration
        elif performance > 0.7:
            self.learning_parameters['exploration_rate'] = min(
                0.5, 
                self.learning_parameters['exploration_rate'] * 1.1
            )  # Increase exploration
    
    def _generate_task(self) -> Dict[str, torch.Tensor]:
        """Generate a learning task based on current capabilities.
        
        Returns:
            Dictionary with 'input', 'target', and 'difficulty'
        """
        # Get current stage characteristics
        stage_char = None
        if self.child and hasattr(self.child, 'curriculum'):
            if hasattr(self.child.curriculum, 'current_stage'):
                stage = self.child.curriculum.current_stage
            if hasattr(self.child.curriculum, 'get_stage_characteristics'):
                stage_char = self.child.curriculum.get_stage_characteristics()
        
        # Generate task difficulty based on current performance
        if self.performance_history:
            avg_performance = np.mean(self.performance_history[-10:])
            difficulty = self.learning_parameters['curriculum_difficulty']
            
            # Adjust difficulty based on performance
            if avg_performance > 0.8:
                difficulty = min(1.0, difficulty + 0.1)
            elif avg_performance < 0.4:
                difficulty = max(0.1, difficulty - 0.1)
                
            self.learning_parameters['curriculum_difficulty'] = difficulty
        else:
            difficulty = self.learning_parameters['curriculum_difficulty']
        
        # Generate input tensor
        embedding_dim = 128  # Default
        if CONFIG_AVAILABLE and hasattr(config, 'embedding_dim'):
            embedding_dim = config.embedding_dim
        
        noise_scale = self.learning_parameters['exploration_rate']
        
        base_input = torch.randn(1, embedding_dim, device=self.device)
        noise = torch.randn_like(base_input) * noise_scale
        task_input = base_input + noise
        
        # Generate target based on stage requirements
        target = self._generate_target(stage_char, embedding_dim)
        
        return {
            'input': task_input,
            'target': target,
            'difficulty': difficulty
        }
    
    def _generate_target(self, stage_char: Optional[Any], target_dim: int) -> torch.Tensor:
        """Generate target tensor based on stage characteristics.
        
        Args:
            stage_char: Stage characteristics (optional)
            target_dim: Target tensor dimension
            
        Returns:
            Target tensor
        """
        if stage_char and hasattr(stage_char, 'complexity_range'):
            complexity = np.interp(
                stage_char.complexity_range[0],
                [0, 1],
                [0.1, 0.9]
            )
        else:
            complexity = 0.5  # Default complexity
        
        # Generate structured target
        target = torch.zeros(1, target_dim, device=self.device)
        num_active = int(target_dim * complexity)
        active_indices = torch.randperm(target_dim, device=self.device)[:num_active]
        target[0, active_indices] = torch.randn(num_active, device=self.device)
        
        return target
    
    def _evaluate_performance(self, 
                            response: torch.Tensor,
                            target: torch.Tensor) -> float:
        """Evaluate the performance of the response against the target.
        
        Args:
            response: Response tensor
            target: Target tensor
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        with torch.no_grad():
            # Ensure tensors are on same device and have compatible shapes
            if response.device != target.device:
                target = target.to(response.device)
            
            # Flatten tensors for comparison
            response_flat = response.flatten()
            target_flat = target.flatten()
            
            # Ensure same length
            min_len = min(len(response_flat), len(target_flat))
            response_flat = response_flat[:min_len]
            target_flat = target_flat[:min_len]
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                response_flat.unsqueeze(0),
                target_flat.unsqueeze(0),
                dim=1
            )
            
            # Scale to [0, 1] range
            performance = (similarity + 1) / 2
            
            return performance.item()
    
    def _adapt_parameters(self, performance: float):
        """Adapt learning parameters based on performance.
        
        Args:
            performance: Performance score (0.0 to 1.0)
        """
        # Adjust learning rate
        if performance < 0.3:
            self.learning_parameters['learning_rate'] *= 0.9
        elif performance > 0.8:
            self.learning_parameters['learning_rate'] *= 1.1
            
        # Clip learning rate
        self.learning_parameters['learning_rate'] = np.clip(
            self.learning_parameters['learning_rate'],
            1e-5,
            1e-2
        )
    
    def reset_learning_parameters(self):
        """Reset learning parameters to default values."""
        default_learning_rate = config.learning_rate if CONFIG_AVAILABLE and hasattr(config, 'learning_rate') else 1e-4
        self.learning_parameters = {
            'learning_rate': default_learning_rate,
            'exploration_rate': 0.3,
            'curriculum_difficulty': 0.1
        }
        self.performance_history = []

    def process_feedback(self, feedback: str, current_stage: Optional[Any] = None, learning_objectives: Optional[Dict[str, Any]] = None) -> None:
        """Process feedback from mother and update learning parameters.
        
        Args:
            feedback: The feedback text from the mother
            current_stage: The current developmental stage (optional)
            learning_objectives: Dictionary of current learning objectives (optional)
        """
        try:
            # Extract learning signals from feedback
            performance = self._evaluate_feedback(feedback)
            
            # Update learning parameters based on performance
            self._adapt_parameters(performance)
            
            # Record performance
            self.performance_history.append(performance)
            
            # Adjust curriculum difficulty based on performance
            if len(self.performance_history) >= 5:
                recent_performance = np.mean(self.performance_history[-5:])
                if recent_performance > 0.8:
                    self.learning_parameters['curriculum_difficulty'] = min(
                        1.0,
                        self.learning_parameters['curriculum_difficulty'] + 0.1
                    )
                elif recent_performance < 0.4:
                    self.learning_parameters['curriculum_difficulty'] = max(
                        0.1,
                        self.learning_parameters['curriculum_difficulty'] - 0.1
                    )
            
            # Decay exploration rate
            self.learning_parameters['exploration_rate'] = max(
                self.min_exploration,
                self.learning_parameters['exploration_rate'] * self.exploration_decay
            )
            
        except Exception as e:
            print(f"Error processing feedback: {str(e)}")
    
    def _evaluate_feedback(self, feedback: str) -> float:
        """Evaluate feedback to determine learning performance.
        
        Args:
            feedback: The feedback text to evaluate
            
        Returns:
            Float between 0 and 1 indicating performance
        """
        # Simple sentiment-based evaluation
        positive_words = {'good', 'great', 'excellent', 'well', 'correct', 'right', 'yes', 'yes!', 'perfect', 'amazing'}
        negative_words = {'bad', 'wrong', 'incorrect', 'no', 'not', "don't", 'stop', 'incorrect', 'try again'}
        
        words = set(feedback.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.5  # Neutral feedback
            
        return positive_count / total_count
    
    def get_performance_statistics(self) -> Dict[str, float]:
        """Get statistics about learning performance.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_history:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        return {
            'mean': float(np.mean(self.performance_history)),
            'std': float(np.std(self.performance_history)),
            'min': float(np.min(self.performance_history)),
            'max': float(np.max(self.performance_history)),
            'count': len(self.performance_history),
            'recent_mean': float(np.mean(self.performance_history[-10:])) if len(self.performance_history) >= 10 else float(np.mean(self.performance_history))
        }

