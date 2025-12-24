#----------------------------------------------------------------------------
#File:       cli.py
#Project:     NeuralChild
#Created by:  Celaya Solutions, 2025
#Author:      Christopher Celaya <chris@chriscelaya.com>
#Description: Command-line interface for NeuralChild
#Version:     1.0.0
#License:     MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

"""Command-line interface for NeuralChild."""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

try:
    from neuralchild import Mind, MotherLLM, Config
    from neuralchild.core import DevelopmentalStage
except ImportError as e:
    print(f"Error: Failed to import NeuralChild modules: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    print("  or")
    print("  pip install -e .")
    print("\nOr run from the NeuralChild directory:")
    print("  python3 -m neuralchild.cli <command>")
    sys.exit(1)


def run_simulation(config_path: str = "config.yaml", steps: int = 100):
    """Run the NeuralChild simulation."""
    print("🧠 NeuralChild - Starting simulation...")
    
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found!")
        return 1
    
    try:
        config = Config.from_yaml(config_path)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    try:
        mind = Mind()
        print(f"✓ Mind created (Stage: {mind.state.developmental_stage.name})")
    except Exception as e:
        print(f"❌ Error creating mind: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        mother = MotherLLM()
        print("✓ Mother LLM initialized")
    except Exception as e:
        print(f"❌ Error creating Mother LLM: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🔄 Running simulation ({steps} steps)...")
    print("-" * 50)
    
    try:
        for step in range(steps):
            # Advance simulation
            mind.step()
            
            # Mother observes and responds (takes Mind object directly)
            response = mother.observe_and_respond(mind)
            
            # Feed response back to mind if there is one
            if response:
                # Convert MotherResponse to input format for mind
                input_data = {
                    "type": "maternal_input",
                    "language": response.response,
                    "action": response.action,
                    "understanding": response.understanding,
                    "development_focus": response.development_focus,
                    "timestamp": response.timestamp.isoformat()
                }
                mind.process_input(input_data)
            
            if step % 10 == 0:
                observable_state = mind.get_observable_state()
                emotions = observable_state.recent_emotions
                print(f"Step {step:3d}: Stage={mind.state.developmental_stage.name}, "
                      f"Emotions={len(emotions)}, Mood={observable_state.apparent_mood:.2f}")
        
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


def show_info():
    """Show system information."""
    print("🧠 NeuralChild - System Information")
    print("=" * 50)
    print(f"Version: 1.0.0")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("\nAvailable Developmental Stages:")
    for stage in DevelopmentalStage:
        print(f"  - {stage.name}")
    print("\nConfiguration:")
    config_path = Path("config.yaml")
    if config_path.exists():
        print(f"  ✓ config.yaml found")
    else:
        print(f"  ✗ config.yaml not found")
    return 0


def init_config(output_path: str = "config.yaml"):
    """Initialize a new configuration file."""
    if os.path.exists(output_path):
        response = input(f"{output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Create default config
    default_config = """# NeuralChild Configuration
server:
  llm_server_url: "http://localhost:1234/v1/chat/completions"
  embedding_server_url: "http://localhost:1234/v1/embeddings"

model:
  llm_model: "gpt-3.5-turbo"
  embedding_model: "all-MiniLM-L6-v2"
  temperature: 0.7
  max_tokens: -1

mind:
  learning_rate: 0.001
  step_interval: 0.1
  starting_stage: "INFANT"
  networks:
    consciousness:
      input_dim: 64
      hidden_dim: 128
      output_dim: 64
    emotions:
      input_dim: 32
      hidden_dim: 64
      output_dim: 32
    perception:
      input_dim: 128
      hidden_dim: 256
      output_dim: 64
    thoughts:
      input_dim: 64
      hidden_dim: 128
      output_dim: 64

logging:
  level: "INFO"
  console_logging: true
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(default_config)
        print(f"✓ Created configuration file: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NeuralChild - A psychological brain simulation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralchild run                    # Run simulation with default settings
  neuralchild run --steps 200        # Run simulation for 200 steps
  neuralchild run --config my.yaml   # Use custom config file
  neuralchild info                   # Show system information
  neuralchild init                   # Create default config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the simulation')
    run_parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    run_parser.add_argument(
        '--steps', '-s',
        type=int,
        default=100,
        help='Number of simulation steps (default: 100)'
    )
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration file')
    init_parser.add_argument(
        '--output', '-o',
        default='config.yaml',
        help='Output path for config file (default: config.yaml)'
    )
    
    # Dashboard command (placeholder - not implemented yet)
    subparsers.add_parser('dashboard', help='Start interactive dashboard (not implemented yet)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'run':
        return run_simulation(args.config, args.steps)
    elif args.command == 'info':
        return show_info()
    elif args.command == 'init':
        return init_config(args.output)
    elif args.command == 'dashboard':
        print("❌ Dashboard not yet implemented.")
        print("See README.md for more information.")
        return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

