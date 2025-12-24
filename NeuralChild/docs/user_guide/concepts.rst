Core Concepts
=============

This section introduces the fundamental concepts and architecture of the NeuralChild framework.
Understanding these concepts will help you effectively use and extend the system.

Architectural Overview
----------------------

NeuralChild is built on a modular architecture that separates concerns while enabling
tight integration between components:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    Mother LLM                           │
   │          (Nurturing, Guidance, Feedback)                │
   └────────────────┬───────────────────────────────────────┘
                    │ Observes & Responds
                    ↓
   ┌─────────────────────────────────────────────────────────┐
   │                      Mind Core                          │
   │  ┌──────────┬──────────┬──────────┬──────────┐         │
   │  │Conscious │ Emotions │Perception│ Thoughts │         │
   │  │  ness    │          │          │          │         │
   │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘         │
   │       └──────────┴──────────┴──────────┘               │
   │              Message Bus                                │
   │  ┌──────────────────────────────────────────────┐      │
   │  │ Memory System │ Belief Network │ Needs       │      │
   │  └──────────────────────────────────────────────┘      │
   └─────────────────────────────────────────────────────────┘

Key Components
--------------

Mind Core
^^^^^^^^^

The :class:`~neuralchild.mind.Mind` class is the central orchestrator of the system. It:

- Manages all neural networks and their lifecycle
- Coordinates communication via the message bus
- Handles memory consolidation and retrieval
- Tracks developmental progress and stage transitions
- Maintains overall state (consciousness, emotions, energy)
- Processes external inputs and generates outputs

**Example:**

.. code-block:: python

   from neuralchild import Mind, Config

   config = Config.from_yaml("config.yaml")
   mind = Mind(config=config)

   # Process input
   mind.process_input({
       "visual": [0.1, 0.2, ...],  # Visual features
       "auditory": [0.3, 0.4, ...],  # Audio features
       "language": "Hello baby!"  # Text input
   })

   # Advance simulation
   mind.step()

   # Get observable state
   state = mind.get_observable_state()

Neural Networks
^^^^^^^^^^^^^^^

Four specialized neural networks work together to simulate cognitive functions:

**Consciousness Network**
   Integrates information from other networks into unified awareness. Maintains a self-model
   and global workspace for conscious processing.

**Emotions Network**
   Processes and generates emotional states using quantum-inspired dynamics. Emotions influence
   memory formation, decision-making, and behavior.

**Perception Network**
   Handles sensory input processing (visual, auditory). Learns patterns and features from
   raw sensory data through experiential learning.

**Thoughts Network**
   Generates cognitive processes, reasoning, and decision-making. Produces internal dialogue
   and action selection.

Each network:

- Grows more complex with developmental stage
- Communicates via typed messages
- Learns from experience through backpropagation
- Has stage-specific activation thresholds

**Example:**

.. code-block:: python

   from neuralchild.mind.networks import (
       ConsciousnessNetwork,
       EmotionsNetwork,
       PerceptionNetwork,
       ThoughtsNetwork
   )

   # Networks are typically created and registered automatically
   # by the Mind, but can be instantiated manually:

   consciousness = ConsciousnessNetwork(
       input_dim=64,
       hidden_dim=128,
       output_dim=64
   )

   # Register with mind
   mind.register_network(consciousness)

Memory System
^^^^^^^^^^^^^

The memory system consists of three levels:

1. **Short-Term Memory (STM)**

   - Limited capacity (recent experiences)
   - Fast access and formation
   - Decays over time if not consolidated
   - Emotionally significant memories strengthen

2. **Long-Term Memory (LTM)**

   - Unlimited capacity
   - Consolidated from STM during "sleep" cycles
   - Organized into semantic clusters
   - Supports recall and pattern matching

3. **Memory Clusters**

   - Semantic groupings of related memories
   - Form higher-level concepts
   - Enable generalization and abstraction
   - Develop with cognitive stage

**Memory Structure:**

.. code-block:: python

   from neuralchild.core import Memory
   from neuralchild.core.schemas import DevelopmentalStage

   memory = Memory(
       id="mem_001",
       content={
           "type": "interaction",
           "data": "Mother said 'good job!'",
           "context": "learning task"
       },
       strength=1.0,
       emotional_valence=0.8,  # Positive emotion
       tags=["language", "praise", "learning"],
       developmental_stage=DevelopmentalStage.TODDLER
   )

   # Memory operations
   memory.access()  # Strengthens memory
   memory.decay(amount=0.01)  # Natural forgetting
   is_forgotten = memory.is_forgotten()  # Check if below threshold

Belief Network
^^^^^^^^^^^^^^

The belief network represents the mind's understanding of the world:

**Belief Structure:**
   Subject-Predicate-Object triples with confidence scores

   Example: "Mother" (subject) "is" (predicate) "helpful" (object)
   Confidence: 0.85

**Features:**

- Bayesian-inspired confidence updates
- Evidence tracking via supporting memories
- Relationship mapping between beliefs
- Contradiction detection
- Natural language generation

**Example:**

.. code-block:: python

   from neuralchild.core import Belief

   belief = Belief(
       subject="ball",
       predicate="is",
       object="round",
       confidence=0.9,
       supporting_memories=["mem_001", "mem_002"]
   )

   # Update with new evidence
   belief.update_confidence(new_evidence=0.95)

   # Convert to natural language
   text = belief.to_natural_language()
   # "I'm sure that ball is round"

Need-Motivation System
^^^^^^^^^^^^^^^^^^^^^^

Drives behavior through simulated physiological and psychological needs:

**Core Needs:**

- **Comfort**: Physical well-being and safety
- **Stimulation**: Cognitive engagement and exploration
- **Rest**: Recovery and memory consolidation
- **Bonding**: Social connection and attachment

**Dynamics:**

- Needs increase over time
- Satisfaction reduces intensity
- Priorities shift by developmental stage
- Expressed when above threshold

**Example:**

.. code-block:: python

   from neuralchild.core import Need

   comfort_need = Need(
       name="comfort",
       intensity=0.7,  # 70% intensity
       satisfaction_level=0.3  # 30% satisfied
   )

   # Update need
   comfort_need.update_intensity(amount=0.1)

   # Satisfy need
   comfort_need.satisfy(amount=0.5)

Message Bus
^^^^^^^^^^^

The :class:`~neuralchild.communication.GlobalMessageBus` enables inter-network communication
using the publish-subscribe pattern:

**Message Types:**

- ``standard``: General information exchange
- ``emotional``: Emotional state broadcasts
- ``belief``: Belief updates and queries
- ``perception``: Sensory input notifications
- ``action``: Action requests and coordination

**Features:**

- Topic-based subscriptions
- Priority-based message filtering
- Developmental stage awareness
- Asynchronous delivery

**Example:**

.. code-block:: python

   from neuralchild.communication import GlobalMessageBus, MessageFilter
   from neuralchild.core import NetworkMessage

   bus = GlobalMessageBus.get_instance()

   # Subscribe to messages
   queue = bus.subscribe(
       subscriber="my_network",
       msg_filter=MessageFilter(
           message_type="emotional",
           min_priority=0.5
       )
   )

   # Publish message
   message = NetworkMessage(
       sender="emotions",
       receiver="consciousness",
       content={"emotion": "joy", "intensity": 0.8},
       message_type="emotional",
       priority=0.9
   )
   bus.publish(message)

   # Retrieve messages
   messages = bus.get_messages(queue, block=False)

Developmental Stages
--------------------

The system progresses through five stages, each with distinct capabilities:

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Stage
     - Age Range
     - Cognitive Abilities
     - Key Features
   * - Infant
     - 0-12 months
     - Basic perception, simple emotions
     - Reflexive, sensory exploration
   * - Toddler
     - 1-3 years
     - Language acquisition, self-awareness
     - Exploration, imitation, simple words
   * - Child
     - 3-12 years
     - Complex reasoning, belief formation
     - Questions, imagination, socialization
   * - Adolescent
     - 12-18 years
     - Abstract thinking, identity
     - Moral reasoning, introspection
   * - Mature
     - 18+ years
     - Wisdom, emotional regulation
     - Metacognition, complex problem-solving

**Progression Criteria:**

Each stage has specific milestones that must be achieved before advancing. See
:doc:`developmental_stages` for detailed information.

Data Flow
---------

The typical data flow through the system:

1. **Input Processing**

   - External stimuli arrive (visual, audio, language)
   - Perception network processes raw input
   - Features extracted and broadcast via message bus

2. **Network Processing**

   - Each network processes relevant messages
   - Emotions network updates emotional state
   - Thoughts network generates cognitive responses
   - Consciousness network integrates information

3. **Memory Formation**

   - Experiences encoded as memories
   - Emotional valence recorded
   - Tagged and stored in short-term memory

4. **State Update**

   - Overall mind state updated
   - Needs updated based on time
   - Energy levels adjusted
   - Developmental progress checked

5. **Output Generation**

   - Observable state compiled
   - Vocalizations generated
   - Behaviors selected
   - Responses prepared for Mother LLM

6. **Mother Interaction**

   - Mother LLM observes state
   - Generates stage-appropriate response
   - Provides guidance and feedback
   - Response fed back to mind

**Visualization:**

.. code-block:: python

   # Complete simulation cycle
   while simulation_running:
       # 1. Mind processes step
       observable_state = mind.step()

       # 2. Mother observes
       if should_respond():
           mother_response = mother.observe_and_respond(observable_state)

           # 3. Feed back to mind
           mind.process_input({
               "type": "maternal_interaction",
               "language": mother_response.response,
               "emotional_tone": mother_response.emotional_tone
           })

       # 4. Check for stage progression
       if mind.check_developmental_progress():
           print(f"Advanced to {mind.current_stage}")

Configuration System
--------------------

The :class:`~neuralchild.config.Config` class provides centralized configuration management:

**Configuration Sections:**

- ``server``: LLM and embedding server URLs
- ``model``: Model selection and parameters
- ``visualization``: Dashboard and display settings
- ``mind``: Neural network dimensions and learning rates
- ``logging``: Logging levels and output
- ``development``: Debug flags and experimental features

**Loading Configuration:**

.. code-block:: python

   from neuralchild import Config, load_config

   # From YAML file
   config = Config.from_yaml("config.yaml")

   # Programmatic configuration
   config = Config()
   config.mind.learning_rate = 0.001
   config.model.temperature = 0.8
   config.development.simulate_llm = True

   # Save configuration
   config.to_yaml("my_config.yaml")

See :doc:`configuration` for complete configuration reference.

Design Principles
-----------------

NeuralChild is built on several key principles:

**Modularity**
   Components are loosely coupled and can be replaced or extended independently.

**Developmental Realism**
   Cognitive progression follows real developmental psychology principles.

**Emergent Behavior**
   Complex behaviors emerge from interactions between simple components.

**Experiential Learning**
   Networks learn from experience rather than explicit training.

**Transparency**
   Internal states are observable and interpretable for research.

**Extensibility**
   New networks, memory types, and behaviors can be added easily.

Next Steps
----------

- Learn about :doc:`developmental_stages` in detail
- Explore :doc:`neural_networks` architecture
- Understand :doc:`mother_llm` interactions
- Configure your simulation: :doc:`configuration`
- Try the :doc:`cli` commands

See Also
--------

- :doc:`../api/core` - Core data structures and schemas
- :doc:`../api/mind` - Mind implementation details
- :doc:`../api/communication` - Message bus API
- :doc:`../examples/index` - Practical examples
