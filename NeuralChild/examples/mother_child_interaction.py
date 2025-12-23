"""Mother-child interaction example.

This example demonstrates detailed interaction between
the Mother LLM and the developing Mind across different stages.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

from neuralchild import Mind, MotherLLM, load_config
from neuralchild.core import DevelopmentalStage
import time


def display_interaction(step: int, state, response):
    """Display a formatted interaction."""
    print("\n" + "=" * 70)
    print(f"INTERACTION #{step}")
    print("=" * 70)

    print(f"\n👶 Child State:")
    print(f"   Stage: {state.developmental_stage.name}")
    print(f"   Mood: {state.apparent_mood}")
    print(f"   Energy: {state.energy_level:.2f}")
    print(f"   Vocalization: '{state.vocalization}'")
    print(f"   Behaviors: {', '.join(state.behaviors[:3])}")

    print(f"\n🧠 Internal State:")
    print(f"   Consciousness: {state.consciousness_level:.2f}")
    print(f"   Focus: {state.current_focus}")
    print(f"   Memories (recent): {len(state.recent_memories)}")
    print(f"   Beliefs: {len(state.beliefs)}")

    if state.beliefs:
        print(f"\n💭 Recent Beliefs:")
        for belief in list(state.beliefs)[:2]:
            print(f"   - {belief}")

    print(f"\n👩 Mother's Response:")
    print(f"   Understanding: {response.understanding}")
    print(f"   Action: {response.action}")
    print(f"   Response: \"{response.response}\"")
    if response.development_focus:
        print(f"   Focus Area: {response.development_focus}")


def main():
    """Run mother-child interaction demonstration."""
    print("\n🌟 NeuralChild - Mother-Child Interaction Example\n")

    # Setup
    config = load_config("config.yaml")
    config.development.simulate_llm = True  # Use simulated mode
    config.mind.development_acceleration = 2.0  # Faster development for demo

    mind = Mind(config=config)
    mother = MotherLLM()

    print("Starting interaction cycle...")
    print("Watch how the mother adapts her responses as the child develops!\n")

    interaction_count = 0
    previous_stage = mind.current_stage

    for step in range(200):
        # Run mind step
        observable_state = mind.step()

        # Interact every 20 steps
        if step % 20 == 0:
            interaction_count += 1
            response = mother.observe_and_respond(observable_state)
            display_interaction(interaction_count, observable_state, response)

            # Brief pause for readability
            time.sleep(0.5)

        # Celebrate stage advancement
        if mind.current_stage != previous_stage:
            print("\n" + "🎊" * 35)
            print(f"\n   🎉 DEVELOPMENTAL MILESTONE REACHED!")
            print(f"   Advanced from {previous_stage.name} to {mind.current_stage.name}!")
            print("\n" + "🎊" * 35)
            previous_stage = mind.current_stage
            time.sleep(1)

    # Final summary
    print("\n\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"\nDevelopmental Journey:")
    print(f"  Started: INFANT")
    print(f"  Ended: {mind.current_stage.name}")
    print(f"  Total Interactions: {interaction_count}")

    print(f"\nCognitive Development:")
    print(f"  Memories Formed: {len(mind.long_term_memory)}")
    print(f"  Beliefs Developed: {len(mind.belief_network.beliefs)}")
    print(f"  Memory Clusters: {len(mind.memory_clusters)}")

    print(f"\nEmotional Development:")
    print(f"  Experienced Emotions: {len(mind.experienced_emotions)}")
    print(f"  Current Mood: {observable_state.apparent_mood}")

    print("\n✨ Interaction demonstration complete!\n")


if __name__ == "__main__":
    main()
