Basic Simulation Example
========================

This example demonstrates a simple NeuralChild simulation.

Full Source
-----------

.. literalinclude:: ../../examples/basic_simulation.py
   :language: python
   :linenos:

Explanation
-----------

This example shows:

1. Loading configuration
2. Creating a Mind instance
3. Creating a Mother LLM
4. Running a simulation loop
5. Observing developmental progress
6. Saving simulation state

Running
-------

.. code-block:: bash

   cd examples
   python basic_simulation.py

Expected Output
---------------

You should see:

* Initialization messages
* Periodic status updates showing developmental stage, mood, and statistics
* Mother's responses to the child's behavior
* Final summary with memories and beliefs formed
* Saved state confirmation

Key Concepts
------------

Observable State
~~~~~~~~~~~~~~~~

The ``observable_state`` contains externally visible information:

* Developmental stage
* Apparent mood
* Vocalizations
* Behaviors
* Recent memories
* Beliefs

This allows external observers (like the Mother LLM) to interact with the mind
without accessing internal state.

Mother-Child Interaction
~~~~~~~~~~~~~~~~~~~~~~~~

The Mother LLM observes the child's state and generates nurturing responses:

.. code-block:: python

   response = mother.observe_and_respond(observable_state)
   print(response.response)  # "You seem happy! Let's play together."

State Persistence
~~~~~~~~~~~~~~~~~

Save and load mind state for long-running simulations:

.. code-block:: python

   # Save
   mind.save_state("models/my_simulation.pt")

   # Load later
   mind = Mind(config=config)
   mind.load_state("models/my_simulation.pt")

Next Steps
----------

* Try :doc:`mother_child_interaction` for a more detailed example
* Read :doc:`../user_guide/concepts` to understand the architecture
* Explore :doc:`../api/index` for full API reference
