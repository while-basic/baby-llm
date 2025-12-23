"""Core Mind class that orchestrates neural networks.

Copyright (c) 2025 Celaya Solutions AI Research Lab

This module implements the central coordinator for all neural networks,
managing communication, development, and the overall state of the artificial mind.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import random
import numpy as np
import torch
import logging
import os
import json
import uuid

from neuralchild.core.schemas import (
    NetworkMessage,
    Memory,
    Belief,
    Need,
    DevelopmentalStage
)
from neuralchild.mind.schemas import (
    MindState,
    ObservableState,
    Emotion,
    EmotionType,
    LanguageAbility
)
from neuralchild.core.neural_network import NeuralNetwork
from neuralchild.communication.message_bus import GlobalMessageBus, MessageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryCluster(BaseModel):
    """Cluster of related memories forming a coherent concept."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    memory_ids: List[str] = Field(default_factory=list)
    centroid: Optional[List[float]] = None
    creation_time: datetime = Field(default_factory=datetime.now)
    last_access_time: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    developmental_stage: DevelopmentalStage = Field(default=DevelopmentalStage.INFANT)

    class Config:
        arbitrary_types_allowed = True

    def access(self) -> None:
        """Update access timestamp."""
        self.last_access_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"id": self.id, "label": self.label, "memory_ids": self.memory_ids,
                "centroid": self.centroid, "creation_time": self.creation_time.isoformat(),
                "last_access_time": self.last_access_time.isoformat(),
                "importance": self.importance, "developmental_stage": self.developmental_stage.name}


class BeliefNetwork(BaseModel):
    """Network of beliefs and their relationships."""
    beliefs: Dict[str, Belief] = Field(default_factory=dict)
    belief_relationships: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    contradictions: List[Tuple[str, str, float]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def add_belief(self, belief: Belief) -> str:
        """Add a belief to the network."""
        belief_id = getattr(belief, 'id', str(uuid.uuid4()))
        if not hasattr(belief, 'id'):
            belief.id = belief_id
        self.beliefs[belief_id] = belief
        if belief_id not in self.belief_relationships:
            self.belief_relationships[belief_id] = {}
        return belief_id

    def update_with_evidence(self, memory_content: Dict[str, Any]) -> List[str]:
        """Update beliefs based on new evidence."""
        affected = []
        for belief_id, belief in self.beliefs.items():
            # Simple relevance check based on content matching
            impact = 0.1 if any(
                str(v) in belief.subject or str(v) in belief.predicate
                for v in memory_content.values() if isinstance(v, str)
            ) else 0.0
            if impact > 0:
                belief.confidence = min(1.0, belief.confidence + impact * 0.05)
                affected.append(belief_id)
        return affected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"beliefs": {k: v.model_dump() for k, v in self.beliefs.items()},
                "relationships": self.belief_relationships, "contradictions": [[a, b, c] for a, b, c in self.contradictions]}


class NeedMotivationSystem(BaseModel):
    """System for managing needs and motivations."""
    needs: Dict[str, Need] = Field(default_factory=dict)
    need_priorities: Dict[str, float] = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.needs:
            self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize default needs."""
        self.needs = {"comfort": Need(name="comfort", intensity=0.7, satisfaction_level=0.3),
                      "stimulation": Need(name="stimulation", intensity=0.8, satisfaction_level=0.2),
                      "rest": Need(name="rest", intensity=0.3, satisfaction_level=0.7),
                      "bonding": Need(name="bonding", intensity=0.9, satisfaction_level=0.1)}
        self.need_priorities = {k: 0.8 for k in self.needs}

    def update_needs(self, elapsed_seconds: float, stage: DevelopmentalStage) -> None:
        """Update need intensities over time."""
        for name, need in self.needs.items():
            rate = 0.0003 * elapsed_seconds
            need.intensity = min(1.0, need.intensity + rate)
            need.satisfaction_level = max(0.0, need.satisfaction_level - 0.0002 * elapsed_seconds)
        self.last_update = datetime.now()

    def get_expressed_needs(self, threshold: float = 0.5) -> Dict[str, float]:
        """Get needs exceeding threshold."""
        return {name: need.intensity for name, need in self.needs.items() if need.intensity > threshold}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"needs": {k: v.model_dump() for k, v in self.needs.items()}, "priorities": self.need_priorities}


class Mind:
    """Core class for the mind simulation orchestrating all neural networks."""

    def __init__(self):
        """Initialize the mind simulation."""
        self.networks: Dict[str, NeuralNetwork] = {}
        self.state = MindState(
            consciousness_level=0.2,
            emotional_state={
                EmotionType.JOY: 0.3,
                EmotionType.TRUST: 0.3,
                EmotionType.FEAR: 0.3,
                EmotionType.SURPRISE: 0.3
            },
            energy_level=0.7,
            developmental_stage=DevelopmentalStage.INFANT,
            language_ability=LanguageAbility(
                vocabulary_size=0,
                sentence_complexity=0.0,
                understanding_level=0.1,
                expression_level=0.0
            )
        )

        # Message bus integration
        self.message_bus = GlobalMessageBus.get_instance()
        self.message_queue = self.message_bus.subscribe(
            "mind",
            MessageFilter(receiver="mind", min_priority=0.3)
        )

        # Memory systems
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self.memory_clusters: Dict[str, MemoryCluster] = {}

        # Belief and need systems
        self.belief_network = BeliefNetwork()
        self.need_system = NeedMotivationSystem()

        # Development tracking
        self.developmental_milestones = {"emotions_experienced": set(), "vocabulary_learned": set(),
                                          "beliefs_formed": 0, "interactions_count": 0, "memories_formed": 0}
        self.development_thresholds = {
            DevelopmentalStage.INFANT: {"emotions_experienced": 3, "interactions_count": 20, "memories_formed": 10},
            DevelopmentalStage.TODDLER: {"emotions_experienced": 5, "vocabulary_learned": 20, "memories_formed": 30},
            DevelopmentalStage.CHILD: {"emotions_experienced": 7, "vocabulary_learned": 100, "beliefs_formed": 10}}

        # Timing
        self.last_consolidation = datetime.now()
        self.last_need_update = datetime.now()
        self.simulation_time = 0.0

        logger.info("Mind initialized at infant developmental stage")

    def register_network(self, network: NeuralNetwork) -> None:
        """Register a neural network with the mind."""
        self.networks[network.name] = network
        network.update_developmental_stage(self.state.developmental_stage)
        logger.info(f"Registered network: {network.name}")

    def process_input(self, input_data: Dict[str, Any]) -> None:
        """Process input from environment."""
        # Process perception inputs
        if "perception" in self.networks and ("visual" in input_data or "auditory" in input_data):
            visual = input_data.get("visual", [0.0] * 64)[:64]
            auditory = input_data.get("auditory", [0.0] * 64)[:64]
            visual += [0.0] * (64 - len(visual))
            auditory += [0.0] * (64 - len(auditory))
            combined = torch.tensor(visual + auditory, dtype=torch.float32)
            self.networks["perception"].experiential_learning(combined)

        # Process language input
        if "language" in input_data and "language" in self.networks:
            text = input_data["language"]
            if isinstance(text, str):
                tokens = text.lower().split()
                self.developmental_milestones["vocabulary_learned"].update(tokens)
                if self.state.developmental_stage.value >= DevelopmentalStage.TODDLER.value and tokens:
                    tensor_data = torch.randn(len(tokens), 10)  # Simplified embedding
                    self.networks["language"].experiential_learning(tensor_data)

        # Form memory
        self._form_memory({"type": input_data.get("type", "input"), "data": input_data, "time": datetime.now().isoformat()})
        self.developmental_milestones["interactions_count"] += 1

    def step(self) -> None:
        """Advance the mind simulation by one step."""
        start_time = datetime.now()

        # Process messages
        messages = self.message_bus.get_messages(self.message_queue, block=False)
        for message in messages:
            self._process_message(message)

        # Update networks
        for network in self.networks.values():
            if hasattr(network, "autonomous_step"):
                network.autonomous_step()

        # Update needs
        elapsed = (datetime.now() - self.last_need_update).total_seconds()
        self.need_system.update_needs(elapsed, self.state.developmental_stage)
        self.last_need_update = datetime.now()

        # Periodic tasks
        self._consolidate_memories()
        self._update_mind_state()
        self._check_developmental_progress()

        # Track time
        self.simulation_time += (datetime.now() - start_time).total_seconds()

    def _form_memory(self, content: Dict[str, Any]) -> None:
        """Form a new memory."""
        memory = Memory(
            content=content,
            importance=0.5,
            emotional_context=dict(self.state.emotional_state),
            developmental_stage=self.state.developmental_stage
        )
        self.short_term_memory.append(memory)
        self.developmental_milestones["memories_formed"] += 1

        # Update beliefs with new evidence
        self.belief_network.update_with_evidence(content)

    def _consolidate_memories(self) -> None:
        """Consolidate short-term memories to long-term."""
        if (datetime.now() - self.last_consolidation).total_seconds() < 60:
            return

        # Move important memories to long-term
        to_consolidate = [m for m in self.short_term_memory if m.importance > 0.6]
        self.long_term_memory.extend(to_consolidate)
        self.short_term_memory = [m for m in self.short_term_memory if m.importance <= 0.6]

        # Simple clustering by developmental stage
        for memory in to_consolidate:
            stage_key = f"cluster_{memory.developmental_stage.name}"
            if stage_key not in self.memory_clusters:
                self.memory_clusters[stage_key] = MemoryCluster(
                    label=f"Memories from {memory.developmental_stage.name}",
                    developmental_stage=memory.developmental_stage
                )
            self.memory_clusters[stage_key].memory_ids.append(memory.id)

        self.last_consolidation = datetime.now()

    def _update_mind_state(self) -> None:
        """Update overall mind state based on subsystems."""
        # Update language ability
        vocab_size = len(self.developmental_milestones["vocabulary_learned"])
        self.state.language_ability = LanguageAbility(
            vocabulary_size=vocab_size,
            sentence_complexity=min(1.0, vocab_size / 500),
            understanding_level=min(1.0, vocab_size / 300),
            expression_level=min(1.0, vocab_size / 400)
        )

        # Update consciousness based on development
        self.state.consciousness_level = 0.2 + (self.state.developmental_stage.value * 0.15)

        # Energy decay
        self.state.energy_level = max(0.1, self.state.energy_level - 0.001)

    def _check_developmental_progress(self) -> None:
        """Check if ready to advance developmental stage."""
        current_stage = self.state.developmental_stage
        if current_stage not in self.development_thresholds:
            return

        thresholds = self.development_thresholds[current_stage]
        ready = True

        for metric, threshold in thresholds.items():
            value = self.developmental_milestones.get(metric, 0)
            if isinstance(value, set):
                value = len(value)
            if value < threshold:
                ready = False
                break

        if ready:
            # Advance to next stage
            next_stage = DevelopmentalStage(current_stage.value + 1)
            self.state.developmental_stage = next_stage
            for network in self.networks.values():
                network.update_developmental_stage(next_stage)
            logger.info(f"Advanced to developmental stage: {next_stage.name}")

    def _process_message(self, message: NetworkMessage) -> None:
        """Process incoming message."""
        if message.message_type == "emotion" and "emotions" in message.content:
            for emotion_str, intensity in message.content["emotions"].items():
                try:
                    emotion_type = EmotionType(emotion_str)
                    self.state.emotional_state[emotion_type] = intensity
                    self.developmental_milestones["emotions_experienced"].add(emotion_type)
                except (ValueError, KeyError):
                    pass

    def get_observable_state(self) -> ObservableState:
        """Get externally observable state of the mind."""
        # Calculate apparent mood
        apparent_mood = sum([
            self.state.emotional_state.get(EmotionType.JOY, 0) * 1.0,
            self.state.emotional_state.get(EmotionType.TRUST, 0) * 0.8,
            self.state.emotional_state.get(EmotionType.SADNESS, 0) * -0.8,
            self.state.emotional_state.get(EmotionType.FEAR, 0) * -0.6,
        ])
        apparent_mood = max(-1.0, min(1.0, apparent_mood))

        # Recent significant emotions
        recent_emotions = [
            Emotion(name=name, intensity=intensity)
            for name, intensity in self.state.emotional_state.items()
            if intensity > 0.2
        ]

        # Vocalization
        vocalization = self._generate_vocalization()

        return ObservableState(
            apparent_mood=apparent_mood,
            energy_level=self.state.energy_level,
            current_focus=self.state.current_focus,
            recent_emotions=recent_emotions,
            expressed_needs=self.need_system.get_expressed_needs(threshold=0.5),
            developmental_stage=self.state.developmental_stage,
            vocalization=vocalization,
            age_appropriate_behaviors=self._get_age_behaviors()
        )

    def _generate_vocalization(self) -> str:
        """Generate age-appropriate vocalization."""
        stage = self.state.developmental_stage
        vocab = list(self.developmental_milestones["vocabulary_learned"])

        if stage == DevelopmentalStage.INFANT:
            return random.choice(["cries", "coos", "babbles", "gurgles"])
        elif stage == DevelopmentalStage.TODDLER and vocab:
            return f"says \"{random.choice(vocab)}\"" if vocab else "babbles"
        elif stage == DevelopmentalStage.CHILD and len(vocab) > 5:
            words = random.sample(vocab, min(3, len(vocab)))
            return f"says \"{' '.join(words)}\""
        return "is quiet"

    def _get_age_behaviors(self) -> List[str]:
        """Get age-appropriate behaviors."""
        behaviors = {
            DevelopmentalStage.INFANT: ["cries when uncomfortable", "responds to voices", "tracks movement"],
            DevelopmentalStage.TODDLER: ["explores environment", "imitates actions", "shows preferences"],
            DevelopmentalStage.CHILD: ["asks questions", "shows imagination", "engages socially"],
        }
        return behaviors.get(self.state.developmental_stage, ["observes"])

    def save_state(self, directory: str = "saved_models") -> bool:
        """Save mind state and all networks."""
        try:
            os.makedirs(directory, exist_ok=True)

            # Save networks
            for name, network in self.networks.items():
                network.save_model(os.path.join(directory, f"{name}.pt"))

            # Save mind state
            state_data = {"developmental_stage": self.state.developmental_stage.value, "consciousness_level": self.state.consciousness_level,
                         "emotional_state": {k.value: v for k, v in self.state.emotional_state.items()}, "energy_level": self.state.energy_level,
                         "simulation_time": self.simulation_time, "milestones": {k: (list(v) if isinstance(v, set) else v) for k, v in self.developmental_milestones.items()},
                         "language": self.state.language_ability.to_dict()}
            with open(os.path.join(directory, "mind_state.json"), "w") as f:
                json.dump(state_data, f, indent=2)

            # Save memories
            memories_data = {"short_term": [m.model_dump(mode='json') for m in self.short_term_memory],
                           "long_term": [m.model_dump(mode='json') for m in self.long_term_memory],
                           "clusters": {k: v.to_dict() for k, v in self.memory_clusters.items()}}
            with open(os.path.join(directory, "memories.json"), "w") as f:
                json.dump(memories_data, f, indent=2)

            # Save beliefs and needs
            with open(os.path.join(directory, "beliefs.json"), "w") as f:
                json.dump(self.belief_network.to_dict(), f, indent=2)
            with open(os.path.join(directory, "needs.json"), "w") as f:
                json.dump(self.need_system.to_dict(), f, indent=2)

            logger.info(f"Mind state saved to {directory}")
            return True

        except Exception as e:
            logger.error(f"Error saving mind state: {e}")
            return False

    def load_state(self, directory: str = "saved_models") -> bool:
        """Load mind state and networks from disk."""
        if not os.path.exists(directory):
            logger.warning(f"Save directory not found: {directory}")
            return False

        try:
            # Load mind state
            with open(os.path.join(directory, "mind_state.json"), "r") as f:
                state_data = json.load(f)
                self.state.developmental_stage = DevelopmentalStage(state_data["developmental_stage"])
                self.state.consciousness_level = state_data["consciousness_level"]
                self.state.emotional_state = {EmotionType(k): v for k, v in state_data["emotional_state"].items()}
                self.state.energy_level = state_data["energy_level"]
                self.simulation_time = state_data["simulation_time"]
                self.developmental_milestones = state_data["milestones"]
                self.state.language_ability = LanguageAbility(**state_data["language"])

                # Convert sets
                if "emotions_experienced" in self.developmental_milestones:
                    self.developmental_milestones["emotions_experienced"] = {
                        EmotionType(e) for e in self.developmental_milestones["emotions_experienced"]
                    }
                if "vocabulary_learned" in self.developmental_milestones:
                    self.developmental_milestones["vocabulary_learned"] = set(self.developmental_milestones["vocabulary_learned"])

            # Load memories
            with open(os.path.join(directory, "memories.json"), "r") as f:
                mem_data = json.load(f)
                self.short_term_memory = [Memory(**m) for m in mem_data["short_term"]]
                self.long_term_memory = [Memory(**m) for m in mem_data["long_term"]]
                self.memory_clusters = {k: MemoryCluster(**v) for k, v in mem_data["clusters"].items()}

            logger.info(f"Mind state loaded from {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading mind state: {e}")
            return False
