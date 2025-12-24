#----------------------------------------------------------------------------
#File:       curriculum_manager.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Curriculum manager for neural child development stages and progress
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Curriculum manager for neural child development.

This module manages the developmental stages and progress of the neural child.

Extracted from neural-child-init/curriculum_manager.py
Adapted imports to use unified structure.
"""

from typing import Dict, Any, Optional
import numpy as np

# Import from unified structure
from neural_child.development.stages import DevelopmentalStage


class CurriculumManager:
    """Manages developmental stages and progress."""

    def __init__(self):
        """Initialize curriculum manager."""
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_progress = 0.0

        # Stage durations in months
        self.stage_durations = {
            DevelopmentalStage.NEWBORN: 3,         # 3 months
            DevelopmentalStage.INFANT: 3,          # 3 months
            DevelopmentalStage.EARLY_TODDLER: 6,   # 6 months
            DevelopmentalStage.LATE_TODDLER: 6,    # 6 months
            DevelopmentalStage.EARLY_PRESCHOOL: 6, # 6 months
            DevelopmentalStage.LATE_PRESCHOOL: 12, # 12 months
            DevelopmentalStage.EARLY_CHILDHOOD: 12, # 12 months
            DevelopmentalStage.MIDDLE_CHILDHOOD: 12, # 12 months
            DevelopmentalStage.LATE_CHILDHOOD: 12, # 12 months
            DevelopmentalStage.PRE_ADOLESCENT: 72, # 72 months (6 years)
            DevelopmentalStage.EARLY_TEEN: 24,     # 24 months (2 years)
            DevelopmentalStage.MID_TEEN: 24,       # 24 months (2 years)
            DevelopmentalStage.LATE_TEEN: 24,      # 24 months (2 years)
            DevelopmentalStage.YOUNG_ADULT: 36,    # 36 months (3 years)
            DevelopmentalStage.EARLY_TWENTIES: 48, # 48 months (4 years)
            DevelopmentalStage.LATE_TWENTIES: 60   # 60 months (5 years)
        }

        # Stage requirements
        self.stage_requirements = {
            DevelopmentalStage.NEWBORN: {
                'interactions': 5,
                'emotional_range': 0.2,
                'vocabulary_size': 3,
                'trust_level': 0.4,
                'q_learning_performance': 0.3
            },
            DevelopmentalStage.INFANT: {
                'interactions': 10,
                'emotional_range': 0.3,
                'vocabulary_size': 8,
                'trust_level': 0.5,
                'q_learning_performance': 0.4
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'interactions': 15,
                'emotional_range': 0.4,
                'vocabulary_size': 15,
                'trust_level': 0.6,
                'q_learning_performance': 0.5
            },
            DevelopmentalStage.LATE_TODDLER: {
                'interactions': 20,
                'emotional_range': 0.5,
                'vocabulary_size': 25,
                'trust_level': 0.7,
                'q_learning_performance': 0.6
            },
            DevelopmentalStage.EARLY_PRESCHOOL: {
                'interactions': 30,
                'emotional_range': 0.6,
                'vocabulary_size': 50,
                'trust_level': 0.75,
                'q_learning_performance': 0.65
            },
            DevelopmentalStage.LATE_PRESCHOOL: {
                'interactions': 40,
                'emotional_range': 0.65,
                'vocabulary_size': 100,
                'trust_level': 0.8,
                'q_learning_performance': 0.7
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                'interactions': 50,
                'emotional_range': 0.7,
                'vocabulary_size': 200,
                'trust_level': 0.85,
                'q_learning_performance': 0.75
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                'interactions': 60,
                'emotional_range': 0.75,
                'vocabulary_size': 400,
                'trust_level': 0.87,
                'q_learning_performance': 0.8
            },
            DevelopmentalStage.LATE_CHILDHOOD: {
                'interactions': 70,
                'emotional_range': 0.8,
                'vocabulary_size': 800,
                'trust_level': 0.9,
                'q_learning_performance': 0.82
            },
            DevelopmentalStage.PRE_ADOLESCENT: {
                'interactions': 100,
                'emotional_range': 0.85,
                'vocabulary_size': 1500,
                'trust_level': 0.92,
                'q_learning_performance': 0.85
            },
            DevelopmentalStage.EARLY_TEEN: {
                'interactions': 150,
                'emotional_range': 0.87,
                'vocabulary_size': 2500,
                'trust_level': 0.93,
                'q_learning_performance': 0.87
            },
            DevelopmentalStage.MID_TEEN: {
                'interactions': 200,
                'emotional_range': 0.9,
                'vocabulary_size': 3500,
                'trust_level': 0.94,
                'q_learning_performance': 0.9
            },
            DevelopmentalStage.LATE_TEEN: {
                'interactions': 250,
                'emotional_range': 0.92,
                'vocabulary_size': 5000,
                'trust_level': 0.95,
                'q_learning_performance': 0.92
            },
            DevelopmentalStage.YOUNG_ADULT: {
                'interactions': 300,
                'emotional_range': 0.94,
                'vocabulary_size': 7500,
                'trust_level': 0.96,
                'q_learning_performance': 0.94
            },
            DevelopmentalStage.EARLY_TWENTIES: {
                'interactions': 400,
                'emotional_range': 0.95,
                'vocabulary_size': 10000,
                'trust_level': 0.97,
                'q_learning_performance': 0.95
            }
        }

        # Progress tracking
        self.interaction_count = 0
        self.unique_emotions = set()
        self.vocabulary_size = 0
        self.trust_level = 0.4
        self.q_learning_performance = 0.0

    def update_progress(
        self,
        interaction_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update progress based on interaction data.

        Args:
            interaction_data: Dictionary containing interaction metrics

        Returns:
            Level up notification if stage advanced, None otherwise
        """
        # Increment interaction count
        self.interaction_count += 1

        # Update vocabulary size if provided
        if 'vocabulary_size' in interaction_data:
            self.vocabulary_size = interaction_data['vocabulary_size']

        # Update trust level if provided
        if ('emotional_state' in interaction_data and
                'trust' in interaction_data['emotional_state']):
            self.trust_level = interaction_data['emotional_state']['trust']

        # Update Q-Learning performance
        if 'q_learning_performance' in interaction_data:
            self.q_learning_performance = interaction_data['q_learning_performance']

        # Get requirements for current stage
        requirements = self.stage_requirements.get(self.current_stage, {})
        if not requirements:
            return None

        # Calculate progress metrics
        progress_metrics = {
            'interactions': min(
                1.0, self.interaction_count / requirements['interactions']
            ),
            'emotional_range': min(
                1.0, len(self.unique_emotions) / 4
            ),  # 4 basic emotions
            'vocabulary_size': min(
                1.0, self.vocabulary_size / requirements['vocabulary_size']
            ),
            'trust_level': min(
                1.0, self.trust_level / requirements['trust_level']
            ),
            'q_learning_performance': min(
                1.0,
                self.q_learning_performance / requirements['q_learning_performance']
            )
        }

        # Calculate overall progress with weights
        weights = {
            'interactions': 0.2,
            'emotional_range': 0.2,
            'vocabulary_size': 0.2,
            'trust_level': 0.2,
            'q_learning_performance': 0.2
        }

        self.stage_progress = sum(
            metric * weights[name]
            for name, metric in progress_metrics.items()
        )

        # Check for stage advancement
        if self.stage_progress >= 1.0:
            return self._advance_stage(progress_metrics)

        return None

    def _advance_stage(
        self,
        progress_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Advance to next developmental stage.

        Args:
            progress_metrics: Current progress metrics

        Returns:
            Level up notification dictionary
        """
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.current_stage)

        if current_idx < len(stages) - 1:
            old_stage = self.current_stage
            self.current_stage = stages[current_idx + 1]
            self.stage_progress = 0.0
            self.interaction_count = 0
            self.unique_emotions = set()

            # Return level up notification with detailed achievements
            return {
                'type': 'level_up',
                'old_stage': old_stage.name,
                'new_stage': self.current_stage.name,
                'message': (
                    f"🎉 Advanced from {old_stage.name} to "
                    f"{self.current_stage.name}!"
                ),
                'achievements': [
                    (
                        f"✨ Mastered {len(self.unique_emotions)} emotions "
                        f"({progress_metrics['emotional_range']*100:.0f}% complete)"
                    ),
                    (
                        f"📚 Learned {self.interaction_count} interactions "
                        f"({progress_metrics['interactions']*100:.0f}% complete)"
                    ),
                    (
                        f"🤝 Built trust to level "
                        f"{progress_metrics['trust_level']*100:.0f}%"
                    ),
                    (
                        f"🔤 Vocabulary size: "
                        f"{progress_metrics['vocabulary_size']*100:.0f}% of goal"
                    ),
                    (
                        f"🧠 Q-Learning performance: "
                        f"{progress_metrics['q_learning_performance']*100:.0f}%"
                    )
                ]
            }

        return None

    def get_current_stage(self) -> DevelopmentalStage:
        """Get current developmental stage.

        Returns:
            Current developmental stage
        """
        return self.current_stage

    def get_stage_progress(self) -> float:
        """Get current stage progress.

        Returns:
            Progress value (0.0 to 1.0)
        """
        return self.stage_progress

    def get_stage_requirements(self) -> Dict[str, Any]:
        """Get requirements for current stage.

        Returns:
            Dictionary of stage requirements
        """
        return self.stage_requirements.get(self.current_stage, {})

