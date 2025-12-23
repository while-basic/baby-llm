"""Basic simulation example for NeuralChild.

This example demonstrates:
- Creating a Mind instance
- Creating a Mother LLM
- Running a simple simulation loop
- Observing developmental progress

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

from neuralchild import Mind, MotherLLM, load_config
from neuralchild.core import DevelopmentalStage


def main():
    """Run a basic NeuralChild simulation."""
    print("🧠 NeuralChild - Basic Simulation Example\n")
    print("=" * 60)

    # Load configuration
    print("\n1️⃣  Loading configuration...")
    config = load_config("config.yaml")

    # Enable simulated LLM mode for this example
    config.development.simulate_llm = True
    print("   ✓ Using simulated LLM mode (no API key required)")

    # Create the Mind
    print("\n2️⃣  Creating Mind...")
    mind = Mind(config=config)
    print(f"   ✓ Mind created at {mind.current_stage.name} stage")

    # Create the Mother LLM
    print("\n3️⃣  Creating Mother LLM...")
    mother = MotherLLM()
    print("   ✓ Mother LLM initialized")

    # Run simulation
    print("\n4️⃣  Running simulation (100 steps)...\n")
    print("-" * 60)

    for step in range(100):
        # Run one mind step
        observable_state = mind.step()

        # Mother observes and responds every 10 steps
        if step % 10 == 0:
            response = mother.observe_and_respond(observable_state)

            # Display progress
            print(f"\n[Step {step:3d}]")
            print(f"  Stage: {observable_state.developmental_stage.name}")
            print(f"  Mood: {observable_state.apparent_mood}")
            print(f"  Vocalization: '{observable_state.vocalization}'")
            print(f"  Memories: {len(observable_state.recent_memories)}")
            print(f"  Beliefs: {len(observable_state.beliefs)}")
            print(f"  Mother says: '{response.response[:60]}...'")

        # Check for developmental advancement
        if step > 0 and step % 20 == 0:
            if mind.current_stage != observable_state.developmental_stage:
                print(f"\n🎉 DEVELOPMENTAL MILESTONE!")
                print(f"   Advanced to {observable_state.developmental_stage.name} stage!")

    # Final summary
    print("\n" + "=" * 60)
    print("\n5️⃣  Simulation Complete!")
    print(f"\n   Final Statistics:")
    print(f"   - Developmental Stage: {mind.current_stage.name}")
    print(f"   - Total Memories: {len(mind.long_term_memory)}")
    print(f"   - Beliefs Formed: {len(mind.belief_network.beliefs)}")
    print(f"   - Emotional State: {list(mind.emotional_state.keys())[:3]}")

    # Save final state
    print("\n6️⃣  Saving final state...")
    mind.save_state("models/basic_simulation_final.pt")
    print("   ✓ State saved to models/basic_simulation_final.pt")

    print("\n✨ Example complete!\n")


if __name__ == "__main__":
    main()
