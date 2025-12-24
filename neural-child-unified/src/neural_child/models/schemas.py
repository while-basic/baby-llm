#----------------------------------------------------------------------------
#File:       schemas.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Unified data models and schemas merged from all projects
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Core schemas for the Neural Child project.

This module defines the data structures and models used throughout the system,
establishing a consistent interface for component interaction.

Merged from:
- NeuralChild/neuralchild/core/schemas.py (base schemas)
- neural-child-init/schemas.py (MotherResponse)
- neural-child-1-main/schemas.py (EmotionalContext, ActionType)
- neural-child-meta-learning-main/schemas.py (MotherResponse variants)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class DevelopmentalStage(Enum):
    """Developmental stages for the artificial mind."""

    INFANT = 1  # 0-12 months equivalent
    TODDLER = 2  # 1-3 years equivalent
    CHILD = 3  # 3-12 years equivalent
    ADOLESCENT = 4  # 12-18 years equivalent
    MATURE = 5  # 18+ years equivalent


class NetworkMessage(BaseModel):
    """Message passed between neural networks.

    These messages form the basis of inter-network communication,
    allowing different parts of the mind to coordinate.
    """

    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str = "standard"  # standard, emotional, belief, etc.
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: float = 1.0  # Higher values indicate higher priority
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "developmental_stage": self.developmental_stage.name,
        }


class NetworkState(BaseModel):
    """State of a neural network.

    Maintains the current state of a neural network, including
    its developmental weights and operational parameters.
    """

    name: str
    active: bool = True
    last_update: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    developmental_weights: Dict[DevelopmentalStage, float] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "name": self.name,
            "active": self.active,
            "last_update": self.last_update.isoformat(),
            "parameters": self.parameters,
            "developmental_weights": {k.name: v for k, v in self.developmental_weights.items()},
        }


class Memory(BaseModel):
    """Memory representation for storing experiences.

    Memories are encoded experiences that can be stored and recalled
    by the mind, forming the basis for learning and development.
    """

    id: str
    content: Dict[str, Any]
    creation_time: datetime = Field(default_factory=datetime.now)
    last_access_time: datetime = Field(default_factory=datetime.now)
    strength: float = 1.0  # How strongly the memory is retained (decays over time)
    emotional_valence: float = 0.0  # Emotional significance (-1 to 1)
    tags: List[str] = Field(default_factory=list)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT

    model_config = {"arbitrary_types_allowed": True}

    def access(self) -> None:
        """Access the memory, updating its strength and access time."""
        self.last_access_time = datetime.now()
        # Memory becomes stronger when accessed (hebbian learning)
        self.strength = min(5.0, self.strength + 0.1)

    def decay(self, amount: float = 0.01) -> None:
        """Decay the memory strength over time."""
        self.strength = max(0.0, self.strength - amount)

    def is_forgotten(self) -> bool:
        """Check if the memory has been forgotten (strength below threshold)."""
        return self.strength < 0.1


class VectorOutput(BaseModel):
    """Vector output from a neural network.

    Represents numerical output from a neural network for
    machine consumption and processing.
    """

    source: str
    data: List[float]
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT

    model_config = {"arbitrary_types_allowed": True}


class TextOutput(BaseModel):
    """Text output from a neural network for human consumption.

    Represents natural language output that can be understood
    by humans interacting with the system.
    """

    source: str
    text: str
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT

    model_config = {"arbitrary_types_allowed": True}


class Belief(BaseModel):
    """Belief representation in the mind.

    Beliefs are mental constructs that represent the mind's
    understanding of concepts and relationships in the world.
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    creation_time: datetime = Field(default_factory=datetime.now)
    last_update_time: datetime = Field(default_factory=datetime.now)
    supporting_memories: List[str] = Field(default_factory=list)  # IDs of supporting memories
    developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANT

    model_config = {"arbitrary_types_allowed": True}

    def update_confidence(self, new_evidence: float) -> None:
        """Update the belief confidence based on new evidence."""
        # Simple Bayesian-inspired update
        self.confidence = (self.confidence + new_evidence) / 2
        self.last_update_time = datetime.now()

    def to_natural_language(self) -> str:
        """Convert the belief to natural language representation."""
        confidence_text = ""
        if self.confidence > 0.8:
            confidence_text = "I'm sure that "
        elif self.confidence > 0.5:
            confidence_text = "I think that "
        else:
            confidence_text = "I'm not sure, but maybe "

        return f"{confidence_text}{self.subject} {self.predicate} {self.object}"


class Need(BaseModel):
    """Representation of a need in the mind.

    Needs drive behavior and motivation, creating
    the impetus for action and learning.
    """

    name: str
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    last_update: datetime = Field(default_factory=datetime.now)
    satisfaction_level: float = Field(ge=0.0, le=1.0, default=0.5)

    model_config = {"arbitrary_types_allowed": True}

    def update_intensity(self, amount: float) -> None:
        """Update the need intensity."""
        self.intensity = max(0.0, min(1.0, self.intensity + amount))
        self.last_update = datetime.now()

    def satisfy(self, amount: float) -> None:
        """Satisfy the need by the specified amount."""
        self.satisfaction_level = min(1.0, self.satisfaction_level + amount)
        # Reduce intensity when satisfied
        self.intensity = max(0.0, self.intensity - (amount * 0.5))
        self.last_update = datetime.now()


class EmotionalContext(BaseModel):
    """Emotional context for interactions and responses."""

    joy: float = Field(default=0.5, ge=0.0, le=1.0)
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    fear: float = Field(default=0.1, ge=0.0, le=1.0)
    surprise: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator('*')
    @classmethod
    def validate_emotion_range(cls, v):
        """Validate emotion values are within range."""
        if not 0 <= v <= 1:
            raise ValueError('Emotional values must be between 0 and 1')
        return v


class ActionType(str, Enum):
    """Types of actions that can be taken in mother-child interactions."""

    FEED = "FEED"
    SLEEP = "SLEEP"
    COMFORT = "COMFORT"
    PLAY = "PLAY"
    TEACH = "TEACH"
    ENCOURAGE = "ENCOURAGE"
    PRAISE = "PRAISE"
    GUIDE = "GUIDE"
    EXPLORE = "EXPLORE"
    REFLECT = "REFLECT"


class MotherResponse(BaseModel):
    """Structured response schema for mother-child interactions.

    Merged from multiple implementations to support all features:
    - Basic response with emotions (neural-child-init)
    - Action types and developmental focus (neural-child-1-main)
    - Simplified variant (neural-child-meta-learning-main)
    """

    content: str = Field(..., min_length=1, max_length=1000, description="The response message")
    emotional_context: EmotionalContext = Field(default_factory=EmotionalContext)
    action: Optional[ActionType] = None
    reward_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Reward score for the interaction")
    success_metric: float = Field(default=0.0, ge=0.0, le=1.0, description="Success metric for the development stage")
    complexity_rating: float = Field(default=0.0, ge=0.0, le=1.0, description="Complexity rating of the interaction")
    self_critique_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Self-awareness score")
    cognitive_labels: List[str] = Field(default_factory=list, description="Cognitive labels for the interaction")
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Effectiveness of the response")
    developmental_focus: Optional[Dict[str, float]] = Field(
        default=None, description="Focus areas for development"
    )

    @field_validator('content')
    @classmethod
    def validate_content_markers(cls, v):
        """Validate content contains action markers (optional)."""
        # Note: Action markers in [BRACKETS] are optional for backward compatibility
        return v

    @field_validator('cognitive_labels')
    @classmethod
    def validate_labels(cls, v):
        """Validate cognitive labels are strings."""
        if not all(isinstance(label, str) for label in v):
            raise ValueError('All cognitive labels must be strings')
        return v

    @field_validator('developmental_focus')
    @classmethod
    def validate_focus(cls, v):
        """Validate developmental focus values."""
        if v is not None:
            if not all(isinstance(k, str) and isinstance(val, float) for k, val in v.items()):
                raise ValueError('Developmental focus must be a dict of string keys and float values')
            if not all(0 <= value <= 1 for value in v.values()):
                raise ValueError('Developmental focus values must be between 0 and 1')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "content": "That's a good attempt! [HUG]",
                "emotional_context": {
                    "joy": 0.8,
                    "trust": 0.6,
                    "fear": 0.05,
                    "surprise": 0.1
                },
                "action": "COMFORT",
                "reward_score": 0.85,
                "success_metric": 0.7,
                "complexity_rating": 0.4,
                "self_critique_score": 0.3,
                "cognitive_labels": ["encouragement", "basic_concept"],
                "effectiveness": 0.75,
                "developmental_focus": {
                    "emotional_regulation": 0.8,
                    "social_skills": 0.6,
                    "cognitive_development": 0.4
                }
            }]
        }
    }

    # Backward compatibility: direct emotion access
    @property
    def joy(self) -> float:
        """Get joy emotion level (backward compatibility)."""
        return self.emotional_context.joy

    @property
    def trust(self) -> float:
        """Get trust emotion level (backward compatibility)."""
        return self.emotional_context.trust

    @property
    def fear(self) -> float:
        """Get fear emotion level (backward compatibility)."""
        return self.emotional_context.fear

    @property
    def surprise(self) -> float:
        """Get surprise emotion level (backward compatibility)."""
        return self.emotional_context.surprise

