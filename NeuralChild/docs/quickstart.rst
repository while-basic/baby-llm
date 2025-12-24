Quickstart Tutorial
===================

This tutorial walks you through creating your first NeuralChild simulation in 10 minutes.

Step 1: Installation
---------------------

Install NeuralChild:

.. code-block:: bash

   pip install neuralchild

Step 2: Create Configuration
-----------------------------

Generate a default configuration file:

.. code-block:: bash

   neuralchild init

This creates ``config.yaml``. For this quickstart, we'll use simulated LLM mode (no API key needed).

Edit ``config.yaml`` and set:

.. code-block:: yaml

   development:
     simulate_llm: true  # Use simulated mode

Step 3: Your First Simulation
------------------------------

Create a file ``my_first_simulation.py``:

.. code-block:: python

   from neuralchild import Mind, MotherLLM, load_config

   # Load configuration
   config = load_config("config.yaml")

   # Create the Mind at INFANT stage
   mind = Mind(config=config)
   print(f"Mind created at {mind.current_stage.name} stage")

   # Create the Mother LLM
   mother = MotherLLM()
   print("Mother LLM initialized")

   # Run 50 simulation steps
   for step in range(50):
       # Mind processes one step
       state = mind.step()

       # Mother observes every 10 steps
       if step % 10 == 0:
           response = mother.observe_and_respond(state)
           print(f"\n[Step {step}]")
           print(f"  Stage: {state.developmental_stage.name}")
           print(f"  Mood: {state.apparent_mood}")
           print(f"  Mother: {response.response[:50]}...")

   print(f"\nFinal stage: {mind.current_stage.name}")
   print(f"Memories: {len(mind.long_term_memory)}")
   print(f"Beliefs: {len(mind.belief_network.beliefs)}")

Run it:

.. code-block:: bash

   python my_first_simulation.py

Step 4: Use the CLI
--------------------

NeuralChild includes a command-line interface:

Run a simulation:

.. code-block:: bash

   neuralchild run --steps 100

View system info:

.. code-block:: bash

   neuralchild info

Step 5: Launch the Dashboard
-----------------------------

Start the interactive dashboard:

.. code-block:: bash

   neuralchild dashboard

Open http://localhost:8050 in your browser to see:

* Real-time developmental stage tracking
* Emotional state visualization
* Memory and belief formation
* Neural network activity

Step 6: Save and Load State
----------------------------

Save the mind's state:

.. code-block:: python

   from neuralchild import Mind, load_config

   config = load_config("config.yaml")
   mind = Mind(config=config)

   # Run simulation
   for step in range(100):
       mind.step()

   # Save state
   mind.save_state("models/my_simulation.pt")
   print("State saved!")

Load it later:

.. code-block:: python

   # Create new mind
   mind = Mind(config=config)

   # Load previous state
   mind.load_state("models/my_simulation.pt")
   print(f"Loaded! Current stage: {mind.current_stage.name}")

   # Continue from where you left off
   for step in range(100):
       mind.step()

Step 7: Explore Examples
-------------------------

Check out the example scripts:

.. code-block:: bash

   cd examples
   python basic_simulation.py
   python mother_child_interaction.py

What's Next?
------------

* **Learn Concepts**: Read :doc:`user_guide/concepts` to understand the architecture
* **Developmental Stages**: Explore :doc:`user_guide/developmental_stages`
* **Neural Networks**: Learn about :doc:`user_guide/neural_networks`
* **Configuration**: Customize :doc:`user_guide/configuration`
* **API Reference**: Dive into :doc:`api/index`

Tips for Success
-----------------

Start Simple
~~~~~~~~~~~~

Begin with simulated LLM mode and short simulations (100-1000 steps).

Monitor Progress
~~~~~~~~~~~~~~~~

Use the dashboard or observable state to track development:

.. code-block:: python

   state = mind.step()
   print(f"Consciousness: {state.consciousness_level:.2f}")
   print(f"Energy: {state.energy_level:.2f}")
   print(f"Behaviors: {state.behaviors}")

Accelerate Development
~~~~~~~~~~~~~~~~~~~~~~

For experiments, speed up developmental progression:

.. code-block:: python

   config.mind.development_acceleration = 5.0  # 5x faster

Save Checkpoints
~~~~~~~~~~~~~~~~

Save periodically during long simulations:

.. code-block:: python

   for step in range(10000):
       mind.step()
       if step % 1000 == 0:
           mind.save_state(f"checkpoint_{step}.pt")

Common Patterns
---------------

Mother-Child Loop
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuralchild import Mind, MotherLLM, load_config

   config = load_config("config.yaml")
   mind = Mind(config=config)
   mother = MotherLLM()

   for step in range(1000):
       state = mind.step()

       if step % 20 == 0:  # Interact every 20 steps
           response = mother.observe_and_respond(state)
           # Use response for logging, analysis, etc.

Developmental Milestone Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   previous_stage = mind.current_stage

   for step in range(5000):
       mind.step()

       # Check for advancement
       if mind.current_stage != previous_stage:
           print(f"Advanced to {mind.current_stage.name}!")
           previous_stage = mind.current_stage

Troubleshooting
---------------

Slow Simulation
~~~~~~~~~~~~~~~

If simulation is slow:

1. Reduce ``step_interval`` in config
2. Decrease network dimensions
3. Use GPU if available

No Development Progress
~~~~~~~~~~~~~~~~~~~~~~~

If the mind isn't advancing stages:

1. Increase ``development_acceleration``
2. Run more steps (some stages require many experiences)
3. Check milestone requirements in the code

Memory Issues
~~~~~~~~~~~~~

If running out of memory:

1. Reduce history size
2. Clear old memories periodically
3. Save and restart simulation

Next Steps
----------

Congratulations! You've completed the quickstart tutorial.

Continue learning:

* :doc:`user_guide/index` - Comprehensive user guide
* :doc:`examples/index` - More code examples
* :doc:`api/index` - Full API documentation
* :doc:`contributing` - Contribute to the project
