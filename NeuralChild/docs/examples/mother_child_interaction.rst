Mother-Child Interaction Example
=================================

This example demonstrates detailed interaction patterns between the Mother LLM
and the developing Mind.

Full Source
-----------

.. literalinclude:: ../../examples/mother_child_interaction.py
   :language: python
   :linenos:

Explanation
-----------

This example shows:

1. Accelerated developmental progression
2. Detailed state monitoring
3. Mother's adaptive responses across stages
4. Developmental milestone celebration
5. Comprehensive statistics tracking

Running
-------

.. code-block:: bash

   cd examples
   python mother_child_interaction.py

Expected Output
---------------

You should see:

* Detailed interaction logs every 20 steps
* Child's internal and external state
* Mother's understanding and responses
* Celebration messages when advancing developmental stages
* Final comprehensive summary

Key Concepts
------------

Developmental Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Speed up development for demonstrations:

.. code-block:: python

   config.mind.development_acceleration = 2.0  # 2x faster

This helps see stage transitions in shorter simulations.

Detailed State Display
~~~~~~~~~~~~~~~~~~~~~~

The example shows how to access and display rich state information:

.. code-block:: python

   print(f"Consciousness: {state.consciousness_level:.2f}")
   print(f"Energy: {state.energy_level:.2f}")
   print(f"Behaviors: {', '.join(state.behaviors[:3])}")

Mother's Adaptive Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Mother LLM adapts responses based on developmental stage:

* **INFANT**: Simple soothing, basic comfort
* **TODDLER**: Encouraging exploration, naming objects
* **CHILD**: Teaching concepts, answering questions
* **ADOLESCENT**: Abstract discussions, guidance
* **MATURE**: Mutual respect, intellectual partnership

Developmental Milestones
~~~~~~~~~~~~~~~~~~~~~~~~

Track when the mind advances to new stages:

.. code-block:: python

   previous_stage = mind.current_stage

   for step in range(5000):
       mind.step()

       if mind.current_stage != previous_stage:
           print(f"Advanced to {mind.current_stage.name}!")
           previous_stage = mind.current_stage

Comprehensive Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

The example demonstrates collecting rich statistics:

* Developmental journey (starting and ending stages)
* Cognitive development (memories, beliefs, clusters)
* Emotional development (experienced emotions, current mood)
* Total interactions

Next Steps
----------

* Create your own custom interaction patterns
* Experiment with different developmental accelerations
* Track specific metrics relevant to your research
* Integrate with external systems using the observable state
