#----------------------------------------------------------------------------
#File:       milestone_tracker.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Comprehensive milestone tracking system for neural child development
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Comprehensive milestone tracking system for neural child development.

Extracted from neural-child-init/milestone_tracker.py
Adapted imports to use unified structure.

Note: LanguageStage dependency will be available in Phase 3.
"""

import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union, TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Import from unified structure
from neural_child.development.stages import DevelopmentalStage

# Optional import for LanguageStage (will be available in Phase 3)
if TYPE_CHECKING:
    from neural_child.cognitive.language.language_development import LanguageStage

try:
    from neural_child.cognitive.language.language_development import LanguageStage
except ImportError:
    LanguageStage = None  # Will be available in Phase 3


@dataclass
class Milestone:
    """Represents a developmental milestone."""

    id: str
    description: str
    stage: DevelopmentalStage
    domain: str
    requirements: Dict[str, float]
    achieved: bool = False
    date_achieved: Optional[str] = None
    progress: float = 0.0


class DomainType(Enum):
    """Types of developmental domains."""

    COGNITIVE = "cognitive"
    LANGUAGE = "language"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    MOTOR = "motor"
    VISION = "vision"
    MEMORY = "memory"
    SELF_AWARENESS = "self_awareness"


class MilestoneTracker:
    """Tracks developmental milestones across all domains."""

    def __init__(self, save_dir: str = "development_results"):
        """Initialize the milestone tracking system.

        Args:
            save_dir: Directory to save milestone progress
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize milestone database
        self.milestones: Dict[str, Milestone] = {}
        self.achieved_milestones: Set[str] = set()

        # Development history
        self.development_history: List[Dict] = []

        # Initialize domain-specific metrics
        self.domain_metrics = {domain: {} for domain in DomainType}

        # Load predefined milestones
        self._initialize_milestones()

    def _initialize_milestones(self) -> None:
        """Initialize predefined developmental milestones."""
        # Cognitive milestones
        self._add_cognitive_milestones()

        # Language milestones
        self._add_language_milestones()

        # Social milestones
        self._add_social_milestones()

        # Emotional milestones
        self._add_emotional_milestones()

        # Motor/Vision milestones
        self._add_vision_milestones()

        # Memory milestones
        self._add_memory_milestones()

        # Self-awareness milestones
        self._add_self_awareness_milestones()

    def _add_cognitive_milestones(self) -> None:
        """Add cognitive development milestones."""
        cognitive_milestones = [
            Milestone(
                id="cog_1",
                description="Basic pattern recognition",
                stage=DevelopmentalStage.NEWBORN,
                domain=DomainType.COGNITIVE.value,
                requirements={"pattern_recognition": 0.3}
            ),
            Milestone(
                id="cog_2",
                description="Object permanence",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.COGNITIVE.value,
                requirements={"object_permanence": 0.5}
            ),
            Milestone(
                id="cog_3",
                description="Causal reasoning",
                stage=DevelopmentalStage.EARLY_TODDLER,
                domain=DomainType.COGNITIVE.value,
                requirements={"causal_understanding": 0.4, "problem_solving": 0.3}
            ),
            Milestone(
                id="cog_4",
                description="Symbolic thinking",
                stage=DevelopmentalStage.LATE_TODDLER,
                domain=DomainType.COGNITIVE.value,
                requirements={"symbolic_representation": 0.4, "imagination": 0.3}
            )
        ]
        for milestone in cognitive_milestones:
            self.milestones[milestone.id] = milestone

    def _add_language_milestones(self) -> None:
        """Add language development milestones."""
        language_milestones = [
            Milestone(
                id="lang_1",
                description="First words",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.LANGUAGE.value,
                requirements={"vocabulary_size": 5, "expression_level": 0.3}
            ),
            Milestone(
                id="lang_2",
                description="Simple sentences",
                stage=DevelopmentalStage.EARLY_TODDLER,
                domain=DomainType.LANGUAGE.value,
                requirements={"grammar_complexity": 0.4, "vocabulary_size": 50}
            ),
            Milestone(
                id="lang_3",
                description="Complex sentences",
                stage=DevelopmentalStage.LATE_TODDLER,
                domain=DomainType.LANGUAGE.value,
                requirements={
                    "grammar_complexity": 0.6,
                    "vocabulary_size": 100,
                    "sentence_structure": 0.4
                }
            ),
            Milestone(
                id="lang_4",
                description="Abstract language concepts",
                stage=DevelopmentalStage.EARLY_PRESCHOOL,
                domain=DomainType.LANGUAGE.value,
                requirements={
                    "abstract_understanding": 0.5,
                    "vocabulary_size": 200,
                    "expression_complexity": 0.5
                }
            )
        ]
        for milestone in language_milestones:
            self.milestones[milestone.id] = milestone

    def _add_social_milestones(self) -> None:
        """Add social development milestones."""
        social_milestones = [
            Milestone(
                id="soc_1",
                description="Basic social awareness",
                stage=DevelopmentalStage.NEWBORN,
                domain=DomainType.SOCIAL.value,
                requirements={"social_interaction": 0.2}
            ),
            Milestone(
                id="soc_2",
                description="Social engagement",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.SOCIAL.value,
                requirements={"social_interaction": 0.4, "emotional_range": 0.3}
            ),
            Milestone(
                id="soc_3",
                description="Peer interaction",
                stage=DevelopmentalStage.EARLY_TODDLER,
                domain=DomainType.SOCIAL.value,
                requirements={
                    "peer_interaction": 0.4,
                    "social_understanding": 0.3,
                    "cooperation": 0.3
                }
            ),
            Milestone(
                id="soc_4",
                description="Complex social dynamics",
                stage=DevelopmentalStage.LATE_TODDLER,
                domain=DomainType.SOCIAL.value,
                requirements={
                    "social_complexity": 0.5,
                    "empathy": 0.4,
                    "group_dynamics": 0.3
                }
            )
        ]
        for milestone in social_milestones:
            self.milestones[milestone.id] = milestone

    def _add_emotional_milestones(self) -> None:
        """Add emotional development milestones."""
        emotional_milestones = [
            Milestone(
                id="emo_1",
                description="Basic emotional expression",
                stage=DevelopmentalStage.NEWBORN,
                domain=DomainType.EMOTIONAL.value,
                requirements={"emotional_range": 0.2}
            ),
            Milestone(
                id="emo_2",
                description="Emotional regulation",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.EMOTIONAL.value,
                requirements={"emotional_regulation": 0.3}
            ),
            Milestone(
                id="emo_3",
                description="Complex emotion recognition",
                stage=DevelopmentalStage.EARLY_TODDLER,
                domain=DomainType.EMOTIONAL.value,
                requirements={
                    "emotion_recognition": 0.4,
                    "emotional_complexity": 0.3
                }
            ),
            Milestone(
                id="emo_4",
                description="Emotional intelligence",
                stage=DevelopmentalStage.LATE_TODDLER,
                domain=DomainType.EMOTIONAL.value,
                requirements={
                    "emotional_intelligence": 0.5,
                    "empathy": 0.4,
                    "self_regulation": 0.4
                }
            )
        ]
        for milestone in emotional_milestones:
            self.milestones[milestone.id] = milestone

    def _add_vision_milestones(self) -> None:
        """Add vision development milestones."""
        vision_milestones = [
            Milestone(
                id="vis_1",
                description="Basic visual tracking",
                stage=DevelopmentalStage.NEWBORN,
                domain=DomainType.VISION.value,
                requirements={"visual_acuity": 0.2}
            ),
            Milestone(
                id="vis_2",
                description="Object recognition",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.VISION.value,
                requirements={"pattern_recognition": 0.3, "visual_acuity": 0.4}
            ),
        ]
        for milestone in vision_milestones:
            self.milestones[milestone.id] = milestone

    def _add_memory_milestones(self) -> None:
        """Add memory development milestones."""
        memory_milestones = [
            Milestone(
                id="mem_1",
                description="Short-term memory formation",
                stage=DevelopmentalStage.NEWBORN,
                domain=DomainType.MEMORY.value,
                requirements={"memory_retention": 0.2}
            ),
            Milestone(
                id="mem_2",
                description="Working memory development",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.MEMORY.value,
                requirements={"memory_retention": 0.4, "attention_span": 0.3}
            ),
        ]
        for milestone in memory_milestones:
            self.milestones[milestone.id] = milestone

    def _add_self_awareness_milestones(self) -> None:
        """Add self-awareness development milestones."""
        self_awareness_milestones = [
            Milestone(
                id="self_1",
                description="Basic self-recognition",
                stage=DevelopmentalStage.INFANT,
                domain=DomainType.SELF_AWARENESS.value,
                requirements={"self_recognition": 0.3}
            ),
            Milestone(
                id="self_2",
                description="Self-concept development",
                stage=DevelopmentalStage.EARLY_TODDLER,
                domain=DomainType.SELF_AWARENESS.value,
                requirements={"self_recognition": 0.5, "emotional_awareness": 0.4}
            ),
        ]
        for milestone in self_awareness_milestones:
            self.milestones[milestone.id] = milestone

    def update_progress(
        self,
        metrics: Dict[str, Union[float, int]],
        stage: DevelopmentalStage
    ) -> Dict[str, List[Milestone]]:
        """Update progress for all milestones based on current metrics.

        Args:
            metrics: Current developmental metrics
            stage: Current developmental stage

        Returns:
            Dict containing lists of achieved and in_progress milestones
        """
        new_achievements = []
        in_progress = []

        # Check all milestones in the given stage
        for milestone_id, milestone in self.milestones.items():
            if (milestone.stage == stage and
                    milestone_id not in self.achieved_milestones):
                progress = self._calculate_milestone_progress(milestone, metrics)

                if progress >= 1.0:
                    new_achievements.append(milestone)
                    self.achieved_milestones.add(milestone_id)
                    self._achieve_milestone(milestone)
                elif progress > 0:
                    milestone.progress = progress
                    in_progress.append(milestone)

        return {
            "new_achievements": new_achievements,
            "in_progress": in_progress
        }

    def _calculate_milestone_progress(
        self,
        milestone: Milestone,
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate progress towards a specific milestone.

        Args:
            milestone: Milestone to calculate progress for
            metrics: Current metrics

        Returns:
            Progress value (0.0 to 1.0)
        """
        if not milestone.requirements:
            return 0.0

        progress_values = []
        for req_name, req_value in milestone.requirements.items():
            if req_name in metrics:
                # Calculate progress as a ratio of current value to required value
                current_value = metrics[req_name]
                progress = current_value / req_value if req_value > 0 else 0.0
                progress_values.append(progress)

        # A milestone is achieved only if all requirements are met
        final_progress = min(progress_values) if progress_values else 0.0
        return final_progress

    def _achieve_milestone(self, milestone: Milestone) -> None:
        """Mark a milestone as achieved.

        Args:
            milestone: Milestone to mark as achieved
        """
        if not milestone.achieved:
            milestone.achieved = True
            milestone.date_achieved = datetime.now().isoformat()
            self.achieved_milestones.add(milestone.id)

    def generate_development_report(self) -> Dict[str, Any]:
        """Generate comprehensive development report.

        Returns:
            Dictionary containing development report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "achieved_milestones": len(self.achieved_milestones),
            "total_milestones": len(self.milestones),
            "progress_by_domain": self._calculate_domain_progress(),
            "recent_achievements": self._get_recent_achievements(5),
            "development_trajectory": self._analyze_development_trajectory(),
            "intervention_suggestions": self.generate_intervention_suggestions()
        }
        return report

    def _calculate_domain_progress(self) -> Dict[str, float]:
        """Calculate progress for each developmental domain.

        Returns:
            Dictionary mapping domain names to progress values
        """
        domain_progress = {}
        for domain in DomainType:
            domain_milestones = [
                m for m in self.milestones.values()
                if m.domain == domain.value
            ]
            if domain_milestones:
                achieved = sum(1 for m in domain_milestones if m.achieved)
                progress = achieved / len(domain_milestones)
                domain_progress[domain.value] = progress
        return domain_progress

    def _get_recent_achievements(self, limit: int) -> List[Dict[str, Any]]:
        """Get most recent milestone achievements.

        Args:
            limit: Maximum number of achievements to return

        Returns:
            List of achievement dictionaries
        """
        achieved = sorted(
            [m for m in self.milestones.values() if m.achieved],
            key=lambda x: x.date_achieved or "",
            reverse=True
        )
        return [
            {
                "id": m.id,
                "description": m.description,
                "domain": m.domain,
                "date_achieved": m.date_achieved
            }
            for m in achieved[:limit]
        ]

    def _analyze_development_trajectory(self) -> Dict[str, Any]:
        """Analyze development trajectory and predict future milestones.

        Returns:
            Dictionary containing trajectory analysis
        """
        # Calculate progress rates
        progress_rates = {}
        for domain in DomainType:
            domain_milestones = [
                m for m in self.milestones.values()
                if m.domain == domain.value
            ]
            if domain_milestones:
                achieved_count = sum(1 for m in domain_milestones if m.achieved)
                total_count = len(domain_milestones)
                progress_rates[domain.value] = achieved_count / total_count

        # Predict next milestones
        next_milestones = self._predict_next_milestones()

        return {
            "progress_rates": progress_rates,
            "predicted_next_milestones": next_milestones
        }

    def _predict_next_milestones(self) -> List[Dict[str, Any]]:
        """Predict the next milestones likely to be achieved.

        Returns:
            List of predicted milestone dictionaries
        """
        unachieved = [
            m for m in self.milestones.values()
            if not m.achieved and m.progress > 0.5
        ]
        sorted_by_progress = sorted(
            unachieved,
            key=lambda x: x.progress,
            reverse=True
        )
        return [
            {
                "id": m.id,
                "description": m.description,
                "domain": m.domain,
                "current_progress": m.progress
            }
            for m in sorted_by_progress[:3]
        ]

    def generate_intervention_suggestions(self) -> List[Dict[str, str]]:
        """Generate suggestions for interventions based on development progress.

        Returns:
            List of intervention suggestion dictionaries
        """
        suggestions = []

        # Analyze each domain for potential interventions
        for domain in DomainType:
            domain_milestones = [
                m for m in self.milestones.values()
                if m.domain == domain.value and not m.achieved
            ]

            if domain_milestones:
                # Find milestones with slow progress
                slow_progress = [
                    m for m in domain_milestones if m.progress < 0.3
                ]
                if slow_progress:
                    suggestion = self._create_intervention_suggestion(
                        domain, slow_progress
                    )
                    suggestions.append(suggestion)

        return suggestions

    def _create_intervention_suggestion(
        self,
        domain: DomainType,
        slow_milestones: List[Milestone]
    ) -> Dict[str, str]:
        """Create specific intervention suggestion for domain.

        Args:
            domain: Domain type
            slow_milestones: List of milestones with slow progress

        Returns:
            Intervention suggestion dictionary
        """
        return {
            "domain": domain.value,
            "concern": f"Slow progress in {len(slow_milestones)} {domain.value} milestones",
            "suggestion": self._get_domain_specific_suggestion(domain, slow_milestones)
        }

    def _get_domain_specific_suggestion(
        self,
        domain: DomainType,
        milestones: List[Milestone]
    ) -> str:
        """Get domain-specific intervention suggestion.

        Args:
            domain: Domain type
            milestones: List of milestones

        Returns:
            Suggestion string
        """
        suggestions = {
            DomainType.COGNITIVE: "Increase pattern recognition and problem-solving activities",
            DomainType.LANGUAGE: "Enhance vocabulary through interactive conversations",
            DomainType.SOCIAL: "Increase social interaction scenarios",
            DomainType.EMOTIONAL: "Focus on emotional recognition and regulation exercises",
            DomainType.MOTOR: "Implement more physical coordination tasks",
            DomainType.VISION: "Introduce more complex visual processing challenges",
            DomainType.MEMORY: "Practice memory retention exercises",
            DomainType.SELF_AWARENESS: "Encourage self-reflection activities"
        }
        return suggestions.get(domain, "Review current development approach")

    def save_progress(self) -> None:
        """Save current progress to a file."""
        progress_data = {
            "achieved_milestones": list(self.achieved_milestones),
            "timestamp": datetime.now().isoformat()
        }

        save_path = self.save_dir / "milestone_progress.json"
        with open(save_path, "w") as f:
            json.dump(progress_data, f, indent=2)

    def load_progress(self) -> None:
        """Load progress from file."""
        load_path = self.save_dir / "milestone_progress.json"
        try:
            with open(load_path, "r") as f:
                progress_data = json.load(f)
                self.achieved_milestones = set(
                    progress_data.get("achieved_milestones", [])
                )
        except FileNotFoundError:
            # Initialize empty progress if file doesn't exist
            self.achieved_milestones = set()

