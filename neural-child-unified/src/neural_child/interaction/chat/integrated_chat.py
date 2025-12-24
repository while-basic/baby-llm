#----------------------------------------------------------------------------
#File:       integrated_chat.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Master chat interface integrating all neural networks and systems
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Master chat interface integrating all neural networks and systems.

Extracted from neural-child-init/integrated_chat.py
Adapted imports to use unified structure.
Many dependencies are optional and will be available in later phases.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import numpy as np

# Optional imports for sentence transformers and ChromaDB
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    print("Warning: sentence-transformers not available. Some features may be limited.")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None
    print("Warning: ChromaDB not available. Memory storage may be limited.")

# Optional imports for unified structure (will be available in later phases)
try:
    from neural_child.emotional.memory import EmotionalMemorySystem, EmotionalMemoryEntry
except ImportError:
    EmotionalMemorySystem = None
    EmotionalMemoryEntry = None
    print("Warning: EmotionalMemorySystem not available. Emotional memory features will be limited.")

try:
    from neural_child.emotional.regulation import EmotionalRegulation
except ImportError:
    EmotionalRegulation = None
    print("Warning: EmotionalRegulation not available. Emotional regulation features will be limited.")

try:
    from neural_child.utils.logger import DevelopmentLogger
except ImportError:
    DevelopmentLogger = None
    print("Warning: DevelopmentLogger not available. Logging features will be limited.")

# Optional imports for modules not yet extracted (will be available in later phases)
try:
    from conversation_system import ConversationSystem
except ImportError:
    ConversationSystem = None
    print("Warning: ConversationSystem not available. Conversation features will be limited.")

try:
    from decision_network import DecisionNetwork
except ImportError:
    DecisionNetwork = None
    print("Warning: DecisionNetwork not available. Decision making features will be limited.")

try:
    from ollama_chat import OllamaChat
except ImportError:
    OllamaChat = None
    print("Warning: OllamaChat not available. LLM integration will be limited.")

try:
    from obsidian_api import ObsidianAPI
except ImportError:
    ObsidianAPI = None
    print("Warning: ObsidianAPI not available. Obsidian integration will be limited.")

try:
    from memory_module import MemoryModule
except ImportError:
    MemoryModule = None
    print("Warning: MemoryModule not available. Memory features will be limited.")

try:
    from self_supervised_trainer import SelfSupervisedTrainer
except ImportError:
    SelfSupervisedTrainer = None
    print("Warning: SelfSupervisedTrainer not available. Training features will be limited.")

try:
    from moral_network import MoralNetwork
except ImportError:
    MoralNetwork = None
    print("Warning: MoralNetwork not available. Moral reasoning features will be limited.")

try:
    from rag_memory import RAGMemorySystem
except ImportError:
    RAGMemorySystem = None
    print("Warning: RAGMemorySystem not available. RAG memory features will be limited.")

try:
    from neural_child.cognitive.metacognition.metacognition_system import SelfAwarenessNetwork
except ImportError:
    try:
        from self_awareness_network import SelfAwarenessNetwork
    except ImportError:
        SelfAwarenessNetwork = None
        print("Warning: SelfAwarenessNetwork not available. Self-awareness features will be limited.")

try:
    from integrated_brain import IntegratedBrain
except ImportError:
    IntegratedBrain = None
    print("Warning: IntegratedBrain not available. Brain processing will be limited.")

try:
    from neural_child.development.stages import DevelopmentalStage
except ImportError:
    try:
        from developmental_stages import DevelopmentalStage
    except ImportError:
        DevelopmentalStage = None
        print("Warning: DevelopmentalStage not available. Development tracking will be limited.")

try:
    from brain_state import BrainState
except ImportError:
    BrainState = None
    print("Warning: BrainState not available. Brain state tracking will be limited.")


class IntegratedChatSystem:
    """Master chat interface integrating all neural networks and systems"""
    
    def __init__(self, initial_stage=None):
        """Initialize the integrated chat system"""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle optional DevelopmentalStage
        if DevelopmentalStage is None:
            # Create a simple enum-like class for stages
            class SimpleStage:
                NEWBORN = "NEWBORN"
            self.stage = initial_stage if initial_stage else SimpleStage.NEWBORN
        else:
            self.stage = initial_stage if initial_stage else DevelopmentalStage.NEWBORN
        
        # Initialize brain state
        if BrainState is not None:
            self.brain_state = BrainState()
        else:
            # Create a simple brain state dict
            self.brain_state = type('BrainState', (), {
                'emotional_valence': 0.5,
                'arousal': 0.3,
                'attention': 0.7,
                'consciousness': 0.8,
                'stress': 0.2,
                'fatigue': 0.1,
                'neurotransmitters': {
                    'dopamine': 0.5,
                    'serotonin': 0.5,
                    'norepinephrine': 0.5,
                    'gaba': 0.5,
                    'glutamate': 0.5
                }
            })()
        
        self.brain_state.emotional_valence = 0.5
        self.brain_state.arousal = 0.3
        self.brain_state.attention = 0.7
        self.brain_state.consciousness = 0.8
        self.brain_state.stress = 0.2
        self.brain_state.fatigue = 0.1
        self.brain_state.neurotransmitters = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'gaba': 0.5,
            'glutamate': 0.5
        }
        
        # Initialize emotional memory parameters
        self.emotional_decay_rate = 0.85
        self.emotional_impact_factor = 0.4
        self.emotional_recovery_rate = 0.1
        self.last_emotion_update = datetime.now()
        
        # Initialize development tracking
        self.emotional_maturity = 0.1
        self.learning_rate = 0.05
        self.max_emotional_change = 0.3
        
        # Initialize brain (if available)
        if IntegratedBrain is not None:
            self.brain = IntegratedBrain(stage=self.stage).to(self.device)
            self.brain.brain_state = self.brain_state
        else:
            self.brain = None
            print("Warning: IntegratedBrain not available. Brain processing disabled.")
        
        # Initialize neural networks (if available)
        if DecisionNetwork is not None:
            self.decision_network = DecisionNetwork().to(self.device)
        else:
            self.decision_network = None
        
        if SelfAwarenessNetwork is not None:
            self.self_awareness = SelfAwarenessNetwork().to(self.device)
        else:
            self.self_awareness = None
        
        if MoralNetwork is not None:
            self.moral_network = MoralNetwork().to(self.device)
        else:
            self.moral_network = None
        
        # Initialize ChromaDB (if available)
        if CHROMADB_AVAILABLE and chromadb is not None:
            persist_dir = Path("chat_memory")
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize collections
            collections_to_init = [
                ("conversation", "Stores conversation history"),
                ("vocabulary", "Stores known words and their embeddings"),
                ("dream", "Stores generated dreams"),
                ("emotional_memory", "Stores emotional memories"),
                ("episodic_memory", "Stores episodic memories"),
                ("semantic_memory", "Stores semantic memories")
            ]
            
            # Delete existing collections
            for collection_name, _ in collections_to_init:
                try:
                    self.chroma_client.delete_collection(collection_name)
                except ValueError:
                    pass
            
            # Create all collections
            for collection_name, description in collections_to_init:
                setattr(
                    self,
                    f"{collection_name}_collection",
                    self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"description": description}
                    )
                )
        else:
            self.chroma_client = None
            print("Warning: ChromaDB not available. Memory collections disabled.")
            # Create dummy collections
            for collection_name in ["conversation", "vocabulary", "dream", 
                                   "emotional_memory", "episodic_memory", "semantic_memory"]:
                setattr(self, f"{collection_name}_collection", None)
        
        # Initialize memory systems (if available)
        if EmotionalMemorySystem is not None:
            self.emotional_memory = EmotionalMemorySystem(
                chat_interface=self,
                sentence_transformer_model="all-MiniLM-L6-v2"
            )
        else:
            self.emotional_memory = None
        
        if RAGMemorySystem is not None:
            self.rag_memory = RAGMemorySystem(
                chat_interface=self,
                sentence_transformer_model="all-MiniLM-L6-v2"
            )
        else:
            self.rag_memory = None
        
        if MemoryModule is not None:
            self.memory_module = MemoryModule()
        else:
            self.memory_module = None
        
        # Initialize logger (if available)
        if DevelopmentLogger is not None:
            self.logger = DevelopmentLogger()
        else:
            self.logger = None
        
        # Initialize language and conversation systems (if available)
        if ConversationSystem is not None and self.logger is not None:
            self.conversation_system = ConversationSystem(logger=self.logger)
        else:
            self.conversation_system = None
        
        if OllamaChat is not None:
            self.ollama = OllamaChat(model="gemma3:1b")  # Use gemma3:1b as specified
        else:
            self.ollama = None
            print("Warning: OllamaChat not available. LLM responses disabled.")
        
        # Initialize emotional regulation (if available)
        if EmotionalRegulation is not None and self.logger is not None:
            self.emotional_regulation = EmotionalRegulation(logger=self.logger)
        else:
            self.emotional_regulation = None
        
        # Initialize self-supervised learning (if available)
        if SelfSupervisedTrainer is not None and self.brain is not None:
            self.self_supervised = SelfSupervisedTrainer(model=self.brain)
        else:
            self.self_supervised = None
        
        # Initialize Obsidian integration (if available)
        if ObsidianAPI is not None:
            obsidian_vault_path = Path("neural_child_vault")
            obsidian_vault_path.mkdir(parents=True, exist_ok=True)
            self.obsidian = ObsidianAPI(vault_path=str(obsidian_vault_path))
        else:
            self.obsidian = None
            print("Warning: ObsidianAPI not available. Obsidian integration disabled.")
        
        # Initialize sentence transformer (if available)
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_transformer = None
            print("Warning: SentenceTransformer not available. Text embeddings disabled.")
        
        # Track conversation state
        self.conversation_history = []
        self.last_interaction = datetime.now()
        self.message_counter = 0
        
        # Initialize development tracking
        self.stage_progress = 0.0
        self.creation_time = datetime.now()
        
        # Create necessary directories
        Path("emotional_memories").mkdir(parents=True, exist_ok=True)
        Path("conversation_history").mkdir(parents=True, exist_ok=True)
        Path("brain_states").mkdir(parents=True, exist_ok=True)
        
        # Initialize dimension projection
        self.input_projection = nn.Linear(384, 512).to(self.device)
        
    def _update_emotional_state(self, message_sentiment: float, message_intensity: float):
        """Update emotional state based on interaction"""
        # Calculate time since last update for decay
        time_delta = (datetime.now() - self.last_emotion_update).total_seconds()
        decay_factor = self.emotional_decay_rate ** (time_delta / 60)
        
        # Get recent emotional history
        recent_states = []
        if len(self.conversation_history) > 0:
            recent_states = [entry['brain_state'] for entry in self.conversation_history[-3:]]
        
        # Calculate emotional volatility
        emotional_volatility = 0.0
        if len(recent_states) >= 2:
            changes = [abs(s['emotional_valence'] - p['emotional_valence']) 
                      for s, p in zip(recent_states[1:], recent_states[:-1])]
            emotional_volatility = sum(changes) / len(changes)
        
        # Calculate base emotional impact with context awareness
        base_impact = message_sentiment * self.emotional_impact_factor
        if message_sentiment > 0:
            if emotional_volatility > 0.3:
                base_impact *= 0.5
            else:
                base_impact *= 1.5
        
        maturity_factor = 1.0 - (self.emotional_maturity * 0.3)
        
        # Update emotional valence with smoother transitions
        if message_sentiment > 0:
            if emotional_volatility < 0.3:
                self.brain_state.emotional_valence = min(0.9, 
                    self.brain_state.emotional_valence * 0.7 + (0.3 + base_impact * maturity_factor))
                self.brain_state.neurotransmitters['dopamine'] = min(0.9,
                    self.brain_state.neurotransmitters['dopamine'] * 0.8 + 0.3)
                self.brain_state.neurotransmitters['serotonin'] = min(0.9,
                    self.brain_state.neurotransmitters['serotonin'] * 0.8 + 0.25)
                self.brain_state.stress = max(0.1,
                    self.brain_state.stress * 0.6)
            else:
                self.brain_state.emotional_valence = min(0.9,
                    self.brain_state.emotional_valence * 0.9 + (0.1 + base_impact * maturity_factor))
                self.brain_state.neurotransmitters['dopamine'] = min(0.9,
                    self.brain_state.neurotransmitters['dopamine'] * 0.9 + 0.1)
                self.brain_state.neurotransmitters['serotonin'] = min(0.9,
                    self.brain_state.neurotransmitters['serotonin'] * 0.9 + 0.1)
                self.brain_state.stress = min(0.9,
                    self.brain_state.stress + 0.1)
        else:
            self.brain_state.emotional_valence = max(0.1,
                self.brain_state.emotional_valence * 0.8 + (base_impact * maturity_factor))
            self.brain_state.neurotransmitters['dopamine'] = max(0.1,
                self.brain_state.neurotransmitters['dopamine'] * 0.9)
            self.brain_state.neurotransmitters['serotonin'] = max(0.1,
                self.brain_state.neurotransmitters['serotonin'] * 0.95)
            self.brain_state.stress = min(0.9,
                self.brain_state.stress * 0.8 + (abs(base_impact) * 0.3))
            
        # Update arousal based on intensity and emotional state
        target_arousal = min(0.9, max(0.1,
            self.brain_state.arousal * decay_factor + 
            message_intensity * 0.3 + 
            emotional_volatility * 0.2 + 
            abs(base_impact) * 0.2))
        self.brain_state.arousal = target_arousal
        
        # Update consciousness more dynamically
        target_consciousness = (
            self.brain_state.arousal +
            self.brain_state.attention +
            self.brain_state.neurotransmitters['dopamine'] +
            (1.0 - self.brain_state.stress) +
            self.brain_state.neurotransmitters['serotonin'] * 0.5 +
            (1.0 - emotional_volatility) * 0.3
        ) / 5.0
        
        self.brain_state.consciousness = min(0.9, max(0.1,
            self.brain_state.consciousness * 0.7 + target_consciousness * 0.3))
        
        # Update emotional maturity with context
        if message_sentiment > 0 and emotional_volatility < 0.3:
            self.emotional_maturity = min(1.0,
                self.emotional_maturity + self.learning_rate * 0.1)
        elif emotional_volatility > 0.3:
            self.emotional_maturity = min(1.0,
                self.emotional_maturity + self.learning_rate * 0.05)
        
        # Update timestamp
        self.last_emotion_update = datetime.now()
        
    def _update_memory_with_emotion(self, message: str, emotion: float, intensity: float):
        """Store memory with emotional context"""
        try:
            # Create flattened metadata for ChromaDB
            brain_state = self.get_brain_state()
            flat_metadata = {
                'age_months': self._get_age_months(),
                'stage': self.stage.name if hasattr(self.stage, 'name') else str(self.stage),
                'emotional_maturity': float(self.emotional_maturity),
                'emotional_valence': float(brain_state['emotional_valence']),
                'arousal': float(brain_state['arousal']),
                'attention': float(brain_state['attention']),
                'consciousness': float(brain_state['consciousness']),
                'stress': float(brain_state['stress']),
                'dopamine': float(brain_state['neurotransmitters']['dopamine']),
                'serotonin': float(brain_state['neurotransmitters']['serotonin']),
                'norepinephrine': float(brain_state['neurotransmitters']['norepinephrine']),
                'intensity': float(abs(intensity)),
                'valence': float(emotion),
                'timestamp': datetime.now().isoformat(),
                'content': message
            }
            
            # Store in ChromaDB emotional memory collection (if available)
            if self.emotional_memory_collection is not None:
                memory_id = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.emotional_memory_collection.add(
                    documents=[message],
                    metadatas=[flat_metadata],
                    ids=[memory_id]
                )
            
            # Store full data structure in Obsidian (if available)
            if self.obsidian is not None:
                self.obsidian.create_note(
                    title=f"Memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    content=f"""# Emotional Memory Entry

## Message
{message}

## Emotional Context
- Valence: {emotion:.2f}
- Intensity: {intensity:.2f}
- Arousal: {self.brain_state.arousal:.2f}

## Brain State
{json.dumps(self.get_brain_state(), indent=2)}

## Metadata
- Age: {self._get_age_months()} months
- Stage: {self.stage.name if hasattr(self.stage, 'name') else str(self.stage)}
- Emotional Maturity: {self.emotional_maturity:.2f}
""",
                    folder="Memories"
                )
            
            # Store in RAG memory (if available)
            if self.rag_memory is not None:
                self.rag_memory.store_memory(
                    content=message,
                    memory_type='emotional',
                    emotional_state=flat_metadata,
                    brain_state=brain_state,
                    metadata=flat_metadata
                )
            
        except Exception as e:
            print(f"Error storing memory: {str(e)}")

    def get_emotional_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Get emotional memories related to query"""
        try:
            if self.emotional_memory_collection is None:
                return []
            
            # Search ChromaDB for relevant memories
            results = self.emotional_memory_collection.query(
                query_texts=[query],
                n_results=min(limit, max(1, len(self.conversation_history)))
            )
            
            memories = []
            if results and results['metadatas']:
                for metadata in results['metadatas'][0]:
                    if isinstance(metadata, dict) and 'content' in metadata:
                        memories.append(metadata)
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []
        
    def process_message(self, message: str) -> str:
        """Process incoming message through all systems"""
        try:
            # Check if required components are available
            if self.sentence_transformer is None:
                return "I'm not working right now. Text embedding system not available."
            
            if self.brain is None:
                return "I'm not working right now. Brain system not available."
            
            if self.ollama is None:
                return "I'm not working right now. LLM system not available."
            
            # Get message embedding and ensure float32 dtype
            message_embedding = self.sentence_transformer.encode(message, convert_to_tensor=True)
            message_embedding = message_embedding.to(self.device).float()
            
            # Process through brain
            projected_input = self.input_projection(message_embedding.unsqueeze(0))
            brain_output = self.brain(projected_input)
            
            # Extract emotional signals with stronger sentiment
            emotional_output = brain_output['emotional'].float()
            sentiment = float(emotional_output[0][0].item()) * 2.0 - 1.0
            intensity = float(emotional_output[0][1].item()) * 1.5
            
            # Update emotional state
            self._update_emotional_state(sentiment, intensity)
            
            # Update memory with emotional context
            self._update_memory_with_emotion(message, sentiment, intensity)
            
            # Record interaction in Obsidian (if available)
            if self.obsidian is not None:
                self.obsidian.create_note(
                    title=f"Interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    content=f"""
# Interaction Record

## Message
{message}

## Emotional State
- Valence: {self.brain_state.emotional_valence:.2f}
- Arousal: {self.brain_state.arousal:.2f}
- Attention: {self.brain_state.attention:.2f}
- Consciousness: {self.brain_state.consciousness:.2f}
- Stress: {self.brain_state.stress:.2f}

## Neurotransmitters
- Dopamine: {self.brain_state.neurotransmitters['dopamine']:.2f}
- Serotonin: {self.brain_state.neurotransmitters['serotonin']:.2f}
- Norepinephrine: {self.brain_state.neurotransmitters['norepinephrine']:.2f}

## Development Stage
- Current Stage: {self.stage.name if hasattr(self.stage, 'name') else str(self.stage)}
- Age (months): {self._get_age_months()}
- Progress: {self.stage_progress:.2f}

## Vocabulary
{self._extract_vocabulary(message)}
"""
                )
            
            # Process through decision network (if available)
            emotional_state = torch.tensor([[
                self.brain_state.emotional_valence,
                self.brain_state.arousal,
                self.brain_state.attention,
                self.brain_state.consciousness
            ]], device=self.device, dtype=torch.float32)
            
            conversation_embeddings = message_embedding.unsqueeze(0).unsqueeze(0)
            memory_context = message_embedding.unsqueeze(0)
            
            if self.decision_network is not None:
                decision_output = self.decision_network(
                    conversation_embeddings,
                    emotional_state,
                    memory_context
                )
            
            # Process through self-awareness network (if available)
            if self.self_awareness is not None:
                self_awareness_output = self.self_awareness(message_embedding.unsqueeze(0))
            
            # Process through moral network (if available)
            if self.moral_network is not None:
                moral_output = self.moral_network(message_embedding.unsqueeze(0))
            
            # Update conversation history with new state
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'brain_state': self.get_brain_state()
            })
            
            # Retrieve relevant memories
            relevant_memories = self.get_emotional_memories(message)
            memory_context = ""
            if relevant_memories:
                memory_context = "\nPrevious relevant memories:\n"
                for memory in relevant_memories:
                    if isinstance(memory, dict) and 'content' in memory:
                        memory_context += f"- {memory['content']}\n"
                        memory_context += f"  (Valence: {memory.get('valence', 0.0):.2f}, "
                        memory_context += f"Trust: {memory.get('serotonin', 0.0):.2f}, "
                        memory_context += f"Fear: {memory.get('stress', 0.0):.2f})\n"
            
            # Generate response using Ollama
            response = self.ollama.generate_response(
                message=message + memory_context,
                emotional_state=self.get_brain_state(),
                brain_state={
                    'sensory': brain_output['sensory'].detach().cpu().numpy().tolist(),
                    'memory': brain_output['memory'].detach().cpu().numpy().tolist(),
                    'emotional': brain_output['emotional'].detach().cpu().numpy().tolist(),
                    'decision': brain_output['decision'].detach().cpu().numpy().tolist()
                },
                stage=self.stage.name if hasattr(self.stage, 'name') else str(self.stage),
                age_months=self._get_age_months()
            )
            
            # Process response through emotional regulation (if available)
            if self.emotional_regulation is not None:
                regulated_response = self.emotional_regulation.regulate_emotion(
                    torch.tensor([
                        self.brain_state.emotional_valence,
                        self.brain_state.arousal,
                        self.brain_state.attention,
                        self.brain_state.consciousness
                    ], device=self.device, dtype=torch.float32)
                )
            
            # Update conversation history with response and final state
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'brain_state': self.get_brain_state()
            })
            
            # Format final response with updated emotional state
            final_response = self._format_response(response)
            
            # Update last interaction time
            self.last_interaction = datetime.now()
            
            return final_response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I'm not working right now. Fix me."
            
    def _format_response(self, response: str) -> str:
        """Format response with emotional state and brain state"""
        if self.brain is None:
            return f"Child: {response}"
        
        brain_state = self.brain.get_brain_state()
        
        formatted = f"Child: {response}\n\n"
        formatted += "Emotional State:\n"
        formatted += f"Joy: {brain_state['emotional_valence']:.2f}\n"
        formatted += f"Trust: {brain_state['neurotransmitters']['serotonin']:.2f}\n"
        formatted += f"Fear: {brain_state['stress']:.2f}\n"
        formatted += f"Arousal: {brain_state['arousal']:.2f}\n"
        formatted += f"Attention: {brain_state['attention']:.2f}\n"
        formatted += f"Consciousness: {brain_state['consciousness']:.2f}\n"
        
        return formatted
        
    def _get_age_months(self) -> int:
        """Calculate age in months since creation"""
        now = datetime.now()
        delta = now - self.creation_time
        return int(delta.days / 30.44)
        
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state"""
        return {
            'emotional_valence': float(self.brain_state.emotional_valence),
            'arousal': float(self.brain_state.arousal),
            'attention': float(self.brain_state.attention),
            'consciousness': float(self.brain_state.consciousness),
            'stress': float(self.brain_state.stress),
            'fatigue': float(self.brain_state.fatigue),
            'neurotransmitters': {
                k: float(v) for k, v in self.brain_state.neurotransmitters.items()
            }
        }
        
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
        
    def save_state(self, path: str = "brain_states"):
        """Save current brain state"""
        if self.brain is None:
            print("Warning: Cannot save state - brain not available")
            return
        
        state = {
            'brain_state': self.brain.get_brain_state(),
            'stage': self.stage.name if hasattr(self.stage, 'name') else str(self.stage),
            'stage_progress': self.stage_progress,
            'creation_time': self.creation_time.isoformat(),
            'last_interaction': self.last_interaction.isoformat()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = Path(path) / f"brain_state_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, state_file: str):
        """Load brain state from file"""
        if DevelopmentalStage is None:
            print("Warning: Cannot load state - DevelopmentalStage not available")
            return
        
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        self.stage = DevelopmentalStage[state['stage']]
        self.stage_progress = state['stage_progress']
        self.creation_time = datetime.fromisoformat(state['creation_time'])
        self.last_interaction = datetime.fromisoformat(state['last_interaction'])
        
        # Update brain state (if brain is available)
        if self.brain is not None:
            self.brain.brain_state.emotional_valence = state['brain_state']['emotional_valence']
            self.brain.brain_state.arousal = state['brain_state']['arousal']
            self.brain.brain_state.attention = state['brain_state']['attention']
            self.brain.brain_state.consciousness = state['brain_state']['consciousness']
            self.brain.brain_state.stress = state['brain_state']['stress']
            self.brain.brain_state.fatigue = state['brain_state']['fatigue']
            
            for nt, value in state['brain_state']['neurotransmitters'].items():
                self.brain.brain_state.neurotransmitters[nt] = value

    def _extract_vocabulary(self, message: str) -> str:
        """Extract and analyze vocabulary from message"""
        words = message.lower().split()
        unique_words = set(words)
        
        # Store in vocabulary collection (if available)
        if self.vocabulary_collection is not None:
            for word in unique_words:
                self.vocabulary_collection.add(
                    documents=[word],
                    metadatas=[{
                        'first_seen': datetime.now().isoformat(),
                        'context': message,
                        'age_months': self._get_age_months(),
                        'stage': self.stage.name if hasattr(self.stage, 'name') else str(self.stage)
                    }],
                    ids=[f"word_{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
                )
        
        return f"""
### New Words Learned
{', '.join(unique_words)}

### Context
{message}
"""

