# 📦 Module Documentation - Neural Child Development System

**Complete documentation of all 52 modules in the unified codebase**

---

## 📋 Table of Contents

1. [Core Systems](#core-systems) (9 modules)
2. [Cognitive Systems](#cognitive-systems) (11 modules)
3. [Emotional Systems](#emotional-systems) (4 modules)
4. [Interaction Systems](#interaction-systems) (6 modules)
5. [Psychological Components](#psychological-components) (3 modules)
6. [Unique Features](#unique-features) (10 modules)
7. [Web Interface](#web-interface) (5 modules)
8. [Visualization](#visualization) (1 module)
9. [Utilities & Models](#utilities--models) (3 modules)

---

## 🧠 Core Systems (9 modules)

### 1. `core/brain/neural_network.py`
**Base neural network class for the mind simulation**

Provides the foundation for all neural networks in the Neural Child project, implementing core functionality for:
- Network development and growth through experience
- Dynamic connection formation and pruning
- Communication protocols between networks
- State management and persistence
- Growth metrics tracking (connection density, activation patterns, learning rate adaptation)

**Key Classes:**
- `NeuralNetwork` - Abstract base class for all neural networks
- `GrowthMetrics` - Tracks network development metrics
- `NeuralGrowthRecord` - Records growth history

**Dependencies:** PyTorch, Pydantic, NumPy

---

### 2. `core/brain/neural_architecture.py`
**Advanced neural architecture mimicking human brain structure**

Implements a biologically-inspired neural architecture with:
- 11 brain regions (prefrontal cortex, temporal lobe, parietal lobe, occipital lobe, limbic system, hippocampus, amygdala, cerebellum, brainstem, thalamus, hypothalamus)
- 10 cognitive functions (attention, memory, learning, reasoning, perception, language, emotion, social, executive, consciousness)
- Neural integration networks connecting regions
- Brain state representation with activity levels and neurotransmitter tracking

**Key Classes:**
- `NeuralArchitecture` - Main architecture class
- `BrainState` - Represents current brain state

**Dependencies:** PyTorch, NumPy

---

### 3. `core/brain/integrated_brain.py`
**Integrated brain architecture combining multiple neural components**

Orchestrates the interaction between multiple neural systems:
- Sensory network for processing inputs
- Memory network for storing and retrieving information
- Emotional network for emotional processing
- Decision network for action selection
- Language development integration
- Heartbeat system integration
- Developmental stage management

**Key Classes:**
- `IntegratedBrain` - Main integrated brain class
- `BrainState` - Current state of the integrated brain
- `DevelopmentalStage` - Enum for developmental stages

**Dependencies:** PyTorch, ChromaDB, optional Q-Learning, Language Development, Heartbeat System

---

### 4. `core/decision/decision_network.py`
**Neural network for decision making in child development AI**

Implements decision-making capabilities with:
- Conversation context encoding using LSTM and attention mechanisms
- Emotional state integration
- Memory context integration
- Developmental stage adaptation
- Learning from feedback
- Multi-head attention for complex decision scenarios

**Key Classes:**
- `ConversationEncoder` - Encodes conversation history
- `DecisionNetwork` - Main decision-making network
- `AttentionMechanism` - Attention-based context processing

**Dependencies:** PyTorch, NumPy

---

### 5. `core/decision/q_learning.py`
**Q-Learning module for neural child development**

Implements reinforcement learning for decision making:
- Q-Network for learning action-value functions
- Experience replay buffer
- Epsilon-greedy exploration strategy
- Bellman equation updates
- Target network for stable learning
- Adaptive learning rate

**Key Classes:**
- `QNetwork` - Q-learning network
- `QLearningSystem` - Complete Q-learning system

**Dependencies:** PyTorch, NumPy

---

### 6. `core/training/training_system.py`
**Comprehensive training system with monitoring, checkpointing, and early stopping**

Manages the training process with:
- Moving average monitoring for metrics
- Checkpoint saving and loading
- Early stopping based on validation metrics
- Learning rate scheduling
- Curriculum-based training integration
- Training history tracking

**Key Classes:**
- `MovingAverageMonitor` - Tracks training metrics
- `TrainingSystem` - Main training orchestrator

**Dependencies:** PyTorch, NumPy, optional CurriculumManager

---

### 7. `core/training/self_supervised_trainer.py`
**Self-supervised learning trainer for neural child development**

Enables learning without explicit labels:
- Masked input generation for self-supervised tasks
- Contrastive learning support
- Prediction tasks (next token, masked reconstruction)
- Training loop management
- Loss computation and optimization

**Key Classes:**
- `SelfSupervisedTrainer` - Self-supervised training system

**Dependencies:** PyTorch

---

### 8. `core/training/replay_system.py`
**Replay system for experience replay and importance sampling**

Manages experience replay for reinforcement learning:
- Experience buffer with capacity management
- Importance sampling weights
- Memory pruning based on importance
- Batch sampling for training
- Decay factor for importance weights

**Key Classes:**
- `ReplayOptimizer` - Replay system with importance sampling

**Dependencies:** PyTorch

---

### 9. `development/stages.py`
**Developmental stages enumeration and management**

Defines the developmental progression:
- 18 developmental stages from NEWBORN to LATE_TWENTIES
- Stage transitions and progress tracking
- Age-to-stage mapping
- Stage-specific capabilities

**Key Classes:**
- `DevelopmentalStage` - Enum for all developmental stages

**Dependencies:** None (pure Python enum)

---

## 🧠 Cognitive Systems (11 modules)

### 10. `cognitive/memory/rag_memory.py`
**Retrieval-Augmented Generation system for neural child memories**

Advanced RAG memory system using ChromaDB:
- Semantic memory storage and retrieval
- Emotional context integration
- Episodic memory support
- Similarity-based search
- Memory consolidation
- Context-aware retrieval

**Key Classes:**
- `RAGMemorySystem` - Main RAG memory system

**Dependencies:** ChromaDB, Sentence Transformers, PyTorch

---

### 11. `cognitive/memory/memory_module.py`
**Memory module with differentiable memory and clustering**

Implements advanced memory management:
- Differentiable memory operations
- Memory clustering for organization
- Importance-based memory retention
- Memory consolidation
- Forgetting mechanisms
- Memory retrieval with similarity search

**Key Classes:**
- `MemoryCluster` - Represents a cluster of related memories
- `MemoryModule` - Main memory management system

**Dependencies:** PyTorch, ReplayOptimizer

---

### 12. `cognitive/memory/memory_store.py`
**Memory store with sentence transformer embeddings**

Persistent memory storage system:
- Sentence transformer embeddings for semantic search
- Memory persistence to disk
- JSON-based storage format
- Memory retrieval by similarity
- Memory tagging and categorization
- Development logger integration

**Key Classes:**
- `MemoryStore` - Persistent memory storage

**Dependencies:** Sentence Transformers, DevelopmentLogger

---

### 13. `cognitive/memory/memory_context.py`
**Memory context for neural child development**

Context representation for memory operations:
- Query context
- Emotional state context
- Brain state context
- Developmental stage context
- Age-based context
- Timestamp tracking

**Key Classes:**
- `MemoryContext` - Context for memory retrieval

**Dependencies:** None (pure Python dataclass)

---

### 14. `cognitive/language/symbol_grounding.py`
**Symbol grounding for neural child development**

Maps concepts to tokens and embeddings:
- Concept-to-token mapping
- Embedding-based symbol representation
- Reverse mapping (embedding to concept)
- Symbol similarity computation
- Vocabulary management

**Key Classes:**
- `SymbolGrounding` - Symbol grounding system

**Dependencies:** PyTorch, text_embed module

---

### 15. `cognitive/language/text_embed.py`
**Text embedding module for neural child development**

Generates text embeddings using transformer models:
- Sentence transformer model initialization
- Text-to-embedding conversion
- Batch processing support
- GPU acceleration
- Normalization options

**Key Functions:**
- `initialize_embedding_model()` - Initialize the embedding model
- `get_embeddings()` - Get embeddings for text

**Dependencies:** Transformers (Hugging Face), PyTorch

---

### 16. `cognitive/vision/vision_development.py`
**Vision development and perception system for the neural child**

Implements visual perception capabilities:
- ResNet50-based visual processing
- Stage-appropriate visual capabilities
- Object recognition
- Visual attention mechanisms
- Image preprocessing
- Visual memory integration

**Key Classes:**
- `VisionDevelopment` - Vision development system

**Dependencies:** PyTorch, TorchVision, PIL (optional), DevelopmentalStage

---

### 17. `cognitive/metacognition/metacognition_system.py`
**Unified metacognition and self-awareness system**

Implements thinking about thinking:
- 7 levels of self-awareness (Physical, Mirror, Emotional, Cognitive, Metacognitive, Social, Abstract)
- Hypothesis generation and testing
- Self-correction mechanisms
- Self-concept graph (NetworkX-based)
- Metacognitive monitoring
- Reflection and introspection

**Key Classes:**
- `MetacognitionSystem` - Main metacognition system
- `SelfAwarenessLevel` - Enum for self-awareness levels

**Dependencies:** PyTorch, NetworkX (optional), JSON

---

### 18. `cognitive/moral/moral_network.py`
**Neural network for moral reasoning and ethical decision making**

Implements moral reasoning capabilities:
- Moral value representation and weighting
- Ethical decision-making
- Moral conflict resolution
- Value-based action selection
- Moral development tracking
- Ethical constraint enforcement

**Key Classes:**
- `MoralNetwork` - Moral reasoning network
- `MoralValue` - Represents a moral value

**Dependencies:** PyTorch

---

## 💭 Emotional Systems (4 modules)

### 19. `emotional/regulation.py`
**Emotional regulation system for neural child development**

Manages emotional state and regulation:
- 4 primary emotions (joy, trust, fear, surprise)
- Complex emotion derivation from primaries
- Emotional regulation mechanisms
- State transitions
- Regulation strategies
- Development logger integration

**Key Classes:**
- `EmotionalState` - Emotional state representation
- `EmotionalRegulation` - Regulation system

**Dependencies:** PyTorch, optional DevelopmentLogger

---

### 20. `emotional/development.py`
**Advanced emotional development and regulation system**

Comprehensive emotional development:
- Emotional capability levels (Basic, Intermediate, Complex)
- Neural networks for emotional processing
- Attachment-based emotional learning
- Emotional memory integration
- Developmental progression
- Stimulus processing and response

**Key Classes:**
- `EmotionalDevelopmentSystem` - Main emotional development system
- `EmotionalCapability` - Enum for emotional capabilities
- `EmotionalState` - Extended emotional state

**Dependencies:** PyTorch, NumPy

---

### 21. `emotional/memory.py`
**Advanced emotional memory system with ChromaDB integration**

Stores and retrieves emotionally-charged memories:
- ChromaDB-based storage
- Emotional context embedding
- Memory retrieval by emotional similarity
- Memory consolidation
- Episodic emotional memories
- Memory decay and forgetting

**Key Classes:**
- `EmotionalMemorySystem` - Main emotional memory system
- `EmotionalMemoryEntry` - Individual memory entry

**Dependencies:** ChromaDB, Sentence Transformers, PyTorch, NumPy

---

### 22. `emotional/embedding.py`
**Emotional embedding with quantum-inspired processing**

Generates emotional embeddings with quantum-inspired features:
- Standard emotional embedding (text-to-emotion mapping)
- Quantum-inspired emotional processing
- Superposition of emotional states
- Entanglement between emotions
- Measurement and collapse operations
- Valence projection

**Key Classes:**
- `EmotionalEmbedder` - Standard embedder
- `QuantumEmotionalEmbedder` - Quantum-inspired embedder

**Dependencies:** PyTorch, optional text_embed module

---

## 💬 Interaction Systems (6 modules)

### 23. `interaction/chat/integrated_chat.py`
**Integrated chat system combining all interaction capabilities**

Unified chat interface integrating:
- Emotional chat capabilities
- Self-awareness chat features
- Memory integration
- Developmental stage adaptation
- Context-aware responses
- Multi-modal interaction support

**Key Classes:**
- `IntegratedChatSystem` - Main integrated chat system

**Dependencies:** Various chat and LLM modules

---

### 24. `interaction/chat/emotional_chat.py`
**Emotional chat system integrating heartbeat responses and memory recording**

Chat system with emotional awareness:
- Heartbeat integration for physiological responses
- Emotional memory recording
- Obsidian integration for logging
- Emotional state tracking during conversations
- Age-appropriate responses
- Emotional context in responses

**Key Classes:**
- `EmotionalChatSystem` - Emotional chat system

**Dependencies:** PyTorch, optional ObsidianAPI, EmotionalMemoryEntry

---

### 25. `interaction/chat/self_awareness_chat.py`
**Interactive chat interface for testing self-awareness network**

Chat system with self-awareness:
- Self-awareness network integration
- Metacognitive responses
- Self-reflection in conversations
- NetworkX-based self-concept visualization
- ChromaDB memory integration
- Sentence transformer embeddings

**Key Classes:**
- `SelfAwarenessChatSystem` - Self-awareness chat system

**Dependencies:** PyTorch, NetworkX (optional), ChromaDB, Sentence Transformers

---

### 26. `interaction/llm/llm_module.py`
**Core LLM integration module for Ollama**

Provides unified interface to Ollama:
- Chat completion function
- Structured output support
- Retry logic with exponential backoff
- Configuration from YAML
- Error handling
- Model management

**Key Functions:**
- `chat_completion()` - Main chat completion function

**Dependencies:** Requests, YAML, JSON

---

### 27. `interaction/llm/ollama_chat.py`
**Chat interface for neural child development using Ollama**

Ollama-specific chat implementation:
- gemma3:1b model configuration
- GPU support
- Conversation history management
- Context-aware prompts
- Response formatting
- Error handling and retries

**Key Classes:**
- `OllamaChat` - Ollama chat interface

**Dependencies:** Requests, PyTorch, YAML, llm_module

---

### 28. `interaction/llm/mother_llm.py`
**Mother LLM component that interacts with the mind simulation**

Implements the "Mother" LLM that guides development:
- Age-appropriate responses
- Developmental stage adaptation
- Emotional support and guidance
- Learning facilitation
- Safety monitoring
- Response validation

**Key Classes:**
- `MotherLLM` - Mother LLM system
- `MotherResponse` - Response structure

**Dependencies:** Pydantic (optional), DevelopmentalStage, llm_module

---

## 🧘 Psychological Components (3 modules)

### 29. `psychological/attachment.py`
**Attachment system for modeling caregiver-child relationships**

Models attachment styles:
- 4 attachment styles (Secure, Anxious, Avoidant, Disorganized)
- Attachment network (neural network)
- Caregiver relationship tracking
- Attachment-based emotional responses
- Relationship history
- Developmental impact

**Key Classes:**
- `AttachmentSystem` - Attachment modeling system

**Dependencies:** PyTorch, optional DevelopmentLogger

---

### 30. `psychological/theory_of_mind.py`
**Theory of Mind system for understanding others' mental states**

Predicts and models others' mental states:
- Mental state prediction network
- Emotional state inference
- Belief and intention modeling
- Attention tracking
- Social context processing
- Perspective-taking

**Key Classes:**
- `TheoryOfMind` - Theory of Mind system

**Dependencies:** PyTorch, optional DevelopmentLogger

---

### 31. `psychological/defense_mechanisms.py`
**Defense mechanisms system for coping with anxiety and stress**

Implements psychological defense mechanisms:
- 7 defense mechanisms (Repression, Projection, Denial, Sublimation, Rationalization, Displacement, Regression)
- Defense mechanism network
- Stress and anxiety detection
- Adaptive defense selection
- Coping strategy activation
- Developmental appropriateness

**Key Classes:**
- `DefenseMechanisms` - Defense mechanisms system

**Dependencies:** PyTorch, optional DevelopmentLogger

---

## ⚡ Unique Features (10 modules)

### 32. `physiological/heartbeat_system.py`
**Heartbeat system for neural child development**

Simulates physiological heart rate:
- Real-time heartbeat based on emotional state
- Neural network modulation
- Memory-triggered responses
- 5 heart rate states (Resting, Elevated, Anxious, Focused, Relaxed)
- Heart rate variability
- Physiological state tracking

**Key Classes:**
- `HeartbeatSystem` - Heartbeat simulation system
- `HeartRateState` - Enum for heart rate states

**Dependencies:** PyTorch, NumPy, optional DevelopmentLogger

---

### 33. `dream/dream_system.py`
**Dream system for neural child development**

Generates and manages dreams:
- Dream generation based on emotional state
- Q-Learning for dream type selection
- Obsidian integration for dream storage
- Dream consolidation
- Emotional processing through dreams
- Memory integration

**Key Classes:**
- `DreamSystem` - Main dream system

**Dependencies:** PyTorch, optional ObsidianAPI, Q-Learning, Transformers

---

### 34. `dream/dream_generator.py`
**Dream generator for creating dream content**

Generates dream narratives:
- LLM-based dream generation
- Emotional context integration
- Age-appropriate dreams
- Dream type classification
- Narrative structure
- Symbolic content

**Key Classes:**
- `DreamGenerator` - Dream generation system

**Dependencies:** LLM module, Sentence Transformers (optional)

---

### 35. `integration/obsidian/obsidian_api.py`
**Python interface to interact with Obsidian vault**

Provides Obsidian vault operations:
- Note creation, reading, updating
- Metadata management
- File operations
- Shell command execution
- Vault structure management
- Link management

**Key Classes:**
- `ObsidianAPI` - Obsidian API interface

**Dependencies:** Path, JSON, YAML (optional)

---

### 36. `integration/obsidian/obsidian_connector.py`
**Obsidian connector for neural child development vault management**

Manages Obsidian vault structure:
- Vault initialization
- Folder structure creation
- Note organization
- Metadata management
- Development logger integration
- YAML frontmatter support

**Key Classes:**
- `ObsidianConnector` - Obsidian vault connector

**Dependencies:** Path, JSON, YAML (optional), DevelopmentLogger (optional)

---

### 37. `integration/obsidian/obsidian_visualizer.py`
**Visualize Obsidian notes and their connections**

Creates visualizations of Obsidian vault:
- NetworkX graph generation
- Note connection mapping
- Graph visualization with Matplotlib
- Relationship analysis
- Link graph construction
- Visual network representation

**Key Classes:**
- `ObsidianVisualizer` - Obsidian visualization system

**Dependencies:** NetworkX (optional), Matplotlib (optional), ObsidianAPI

---

### 38. `integration/obsidian/obsidian_heartbeat_logger.py`
**Obsidian integration for heartbeat logging and memory storage**

Logs heartbeat events to Obsidian:
- Heartbeat event logging
- Memory storage in Obsidian
- Statistics tracking
- Timestamp management
- Emotional context logging
- Vault organization

**Key Classes:**
- `ObsidianHeartbeatLogger` - Heartbeat logger for Obsidian

**Dependencies:** ObsidianAPI, Path, JSON

---

### 39. `communication/message_bus.py`
**Message bus for communication between neural networks**

Centralized message routing system:
- Publish-subscribe pattern
- Message prioritization
- Filtering and routing
- Thread-safe operations
- Message queuing
- Priority-based delivery

**Key Classes:**
- `MessageBus` - Main message bus system
- `MessageFilter` - Message filtering configuration

**Dependencies:** Pydantic, Threading, Queue, optional NetworkMessage

---

### 40. `learning/autonomous_learner.py`
**Autonomous learning system for curiosity-driven self-directed learning**

Enables independent learning:
- Self-directed learning task generation
- Self-evaluation of performance
- Adaptive learning parameters
- Dynamic curriculum adjustment
- Curiosity-driven exploration
- Learning goal setting

**Key Classes:**
- `AutonomousLearner` - Autonomous learning system

**Dependencies:** PyTorch, NumPy, optional Config

---

### 41. `safety/safety_monitor.py`
**Safety monitor for harm detection and ethical constraints**

Ensures safe operation:
- Harm detection
- Ethical constraint enforcement
- Age-appropriateness checking
- Content filtering
- Safety exception raising
- Ethical guidelines compliance

**Key Classes:**
- `SafetyMonitor` - Safety monitoring system
- `SafetyException` - Exception for safety violations

**Dependencies:** Regex, optional DevelopmentalStage, EthicalConstraintsConfig

---

## 🌐 Web Interface (5 modules)

### 42. `web/app.py`
**Flask web application for neural child development system**

Main web application with:
- RESTful API endpoints
- System state management
- Chat interface integration
- Visualization data endpoints
- Development control endpoints
- Watch mode for automatic monitoring
- Auto-refresh functionality
- Stage progression alerts

**Key Functions:**
- `create_app()` - Flask application factory
- `app_factory` - Alias for compatibility

**Dependencies:** Flask, Flask-CORS, optional LLM modules

---

### 43. `web/templates/index.html`
**Main HTML template for the web dashboard**

Web interface template with:
- Real-time dashboard
- Emotional state visualization
- Developmental metrics display
- Chat interface
- Progress tracking
- Stage progression indicators
- Alert system

**Dependencies:** Jinja2 templating

---

### 44. `web/templates/base.html`
**Base HTML template**

Base template providing:
- Common layout structure
- CSS and JavaScript includes
- Navigation structure
- Footer with attribution

**Dependencies:** Jinja2 templating

---

### 45. `web/static/css/style.css`
**CSS styles for the web interface**

Styling for:
- Dashboard layout
- Emotional state displays
- Progress bars
- Alert notifications
- Watch mode indicators
- Responsive design

**Dependencies:** None (pure CSS)

---

### 46. `web/static/js/app.js`
**JavaScript for client-side interactions**

Client-side functionality:
- API communication
- Real-time updates
- Chat interface
- Dashboard updates
- Alert handling
- Progress tracking

**Dependencies:** None (vanilla JavaScript)

---

## 📊 Visualization (1 module)

### 47. `visualization/visualization.py`
**Module for visualizing emotional states and neural network architectures**

Creates visualizations:
- Emotional state heatmaps
- Emotional timelines
- Network architecture diagrams
- Psychological metrics visualization
- Learning curves
- NetworkX graph visualizations

**Key Functions:**
- Various plotting functions for different visualization types

**Dependencies:** Matplotlib (optional), Seaborn (optional), NetworkX (optional), PyTorch (optional), Plotly (optional)

---

## 🛠️ Utilities & Models (3 modules)

### 48. `models/schemas.py`
**Unified data models and schemas**

Defines all data structures:
- DevelopmentalStage enum
- NetworkMessage schema
- NetworkState schema
- VectorOutput schema
- TextOutput schema
- EmotionalContext schema
- ActionType enum
- MotherResponse schema

**Key Classes:**
- Various Pydantic models for data validation

**Dependencies:** Pydantic, Enum, DateTime

---

### 49. `utils/logger.py`
**Development logger for the neural child system**

Logging system:
- Development event logging
- Interaction logging
- Error logging
- Vision logging
- Training logging
- JSON-based log storage

**Key Classes:**
- `DevelopmentLogger` - Main logging system

**Dependencies:** Logging, JSON, Path

---

### 50. `utils/helpers.py`
**Helper utility functions**

Utility functions:
- LLM response parsing
- JSON parsing with fallbacks
- Data validation
- Type conversion
- Error handling utilities
- Common operations

**Key Functions:**
- `parse_llm_response()` - Parse LLM responses
- Various helper utilities

**Dependencies:** JSON, PyTorch, NumPy

---

### 51. `development/curriculum_manager.py`
**Curriculum manager for neural child development stages and progress**

Manages developmental curriculum:
- Stage progression tracking
- Stage duration management
- Progress calculation
- Milestone tracking
- Learning objectives per stage
- Curriculum adaptation

**Key Classes:**
- `CurriculumManager` - Curriculum management system

**Dependencies:** NumPy, DevelopmentalStage

---

### 52. `development/milestone_tracker.py`
**Comprehensive milestone tracking system for neural child development**

Tracks developmental milestones:
- Milestone definition and tracking
- Domain-specific milestones (Cognitive, Emotional, Social, Physical, Language)
- Milestone achievement detection
- Progress reporting
- Milestone history
- Stage-appropriate milestones

**Key Classes:**
- `MilestoneTracker` - Milestone tracking system
- `Milestone` - Individual milestone representation
- `DomainType` - Enum for milestone domains

**Dependencies:** PyTorch, JSON, NumPy, DevelopmentalStage, optional LanguageStage

---

## 📝 Additional Files

### `__main__.py`
**Main entry point for the neural child system**

CLI interface with:
- Command-line argument parsing
- Web interface launching
- Watch mode support
- Development mode options
- System initialization

**Dependencies:** argparse, various system modules

---

### `version.py`
**Version information**

Contains version number and metadata for the package.

---

## 🔗 Module Dependencies Overview

### Core Dependencies (Required)
- **PyTorch** - Neural network framework
- **NumPy** - Numerical computing
- **Pydantic** - Data validation
- **JSON** - Data serialization

### Optional Dependencies
- **ChromaDB** - Vector database for memory
- **Sentence Transformers** - Text embeddings
- **Transformers** - Hugging Face models
- **NetworkX** - Graph visualization
- **Matplotlib/Seaborn/Plotly** - Plotting
- **Flask** - Web framework
- **Obsidian API** - Obsidian integration
- **YAML** - Configuration files

---

## 📊 Module Statistics

- **Total Modules:** 52
- **Core Systems:** 9
- **Cognitive Systems:** 11
- **Emotional Systems:** 4
- **Interaction Systems:** 6
- **Psychological Components:** 3
- **Unique Features:** 10
- **Web Interface:** 5
- **Visualization:** 1
- **Utilities & Models:** 3

---

## 🎯 Module Categories by Function

### Neural Networks (15 modules)
- All core/brain modules
- Decision networks
- Emotional networks
- Cognitive networks
- Psychological networks

### Memory Systems (5 modules)
- RAG memory
- Memory module
- Memory store
- Memory context
- Emotional memory

### Learning Systems (4 modules)
- Training system
- Self-supervised trainer
- Replay system
- Autonomous learner

### Interaction Systems (6 modules)
- All chat systems
- All LLM modules

### Integration Systems (4 modules)
- All Obsidian modules

### Development Systems (3 modules)
- Stages
- Curriculum manager
- Milestone tracker

### Safety & Monitoring (2 modules)
- Safety monitor
- Development logger

### Visualization & Web (6 modules)
- Visualization
- Web app
- Templates and static files

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

*Last Updated: January 2025*

