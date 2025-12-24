#!/usr/bin/env python3
#----------------------------------------------------------------------------
#File:       run_neuralchild.py
#Project:     NeuralChild
#Created by:  Celaya Solutions, 2025
#Author:      Christopher Celaya <chris@chriscelaya.com>
#Description: Simple runner script for NeuralChild simulation
#Version:     1.0.0
#License:     MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

"""Simple runner script for NeuralChild simulation."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from neuralchild import Mind, MotherLLM, Config
    from neuralchild.core import DevelopmentalStage
    
    def main():
        """Run the NeuralChild simulation."""
        print("🧠 NeuralChild - Starting simulation...")
        
        # Load configuration
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Error: {config_path} not found!")
            return 1
        
        try:
            config = Config.from_yaml(config_path)
            print("✓ Configuration loaded")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return 1
        
        # Create the mind
        try:
            mind = Mind(config=config)
            print(f"✓ Mind created (Stage: {mind.current_stage.name})")
        except Exception as e:
            print(f"❌ Error creating mind: {e}")
            return 1
        
        # Create the mother LLM
        try:
            mother = MotherLLM()
            print("✓ Mother LLM initialized")
        except Exception as e:
            print(f"❌ Error creating Mother LLM: {e}")
            return 1
        
        # Run simulation
        print("\n🔄 Running simulation (100 steps)...")
        print("-" * 50)
        
        try:
            for step in range(100):
                # Process a simulation step
                observable_state = mind.step()
                
                # Mother observes and responds
                response = mother.observe_and_respond(observable_state)
                
                # Feed response back to mind
                mind.receive_maternal_input(response)
                
                if step % 10 == 0:
                    print(f"Step {step:3d}: Stage={mind.current_stage.name}, "
                          f"Emotions={len(observable_state.get('emotions', []))}")
            
            print("-" * 50)
            print("✓ Simulation completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Simulation interrupted by user")
            return 0
        except Exception as e:
            print(f"\n❌ Error during simulation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    print("  or")
    print("  pip install -e .")
    sys.exit(1)

