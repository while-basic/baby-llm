"""Command-line interface for NeuralChild.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from typing import Optional
import logging

from neuralchild import __version__, Mind, MotherLLM, load_config, get_config
from neuralchild.core import DevelopmentalStage

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Manages the simulation lifecycle."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the simulation runner.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.mind: Optional[Mind] = None
        self.mother: Optional[MotherLLM] = None
        self.running = False
        self.step_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\n🛑 Shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def initialize(self):
        """Initialize the Mind and Mother LLM."""
        logger.info("Initializing NeuralChild simulation...")
        print("🧠 Initializing Mind...")
        self.mind = Mind(config=self.config)

        print("👩 Initializing Mother LLM...")
        self.mother = MotherLLM()

        logger.info("Initialization complete")
        print("✅ Initialization complete!\n")

    def run(self, max_steps: Optional[int] = None, save_interval: int = 100):
        """Run the simulation.

        Args:
            max_steps: Maximum number of steps (None for infinite)
            save_interval: Save state every N steps
        """
        if self.mind is None or self.mother is None:
            self.initialize()

        self.running = True
        print("🚀 Starting simulation...\n")
        print("Press Ctrl+C to stop gracefully\n")
        print("=" * 60)

        try:
            while self.running and (max_steps is None or self.step_count < max_steps):
                # Run one simulation step
                observable_state = self.mind.step()
                self.step_count += 1

                # Mother observes and responds
                if self.step_count % 10 == 0:  # Respond every 10 steps
                    response = self.mother.observe_and_respond(observable_state)
                    # Feed response back to mind (future enhancement)
                    logger.debug(f"Mother response: {response.response}")

                # Display progress
                if self.step_count % 10 == 0:
                    self._display_status(observable_state)

                # Save state periodically
                if self.step_count % save_interval == 0:
                    self._save_checkpoint()

                # Sleep to control simulation speed
                time.sleep(self.config.mind.step_interval)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self.stop()

    def _display_status(self, state):
        """Display current simulation status.

        Args:
            state: Observable state from the mind
        """
        print(f"\r[Step {self.step_count:6d}] "
              f"Stage: {state.developmental_stage.name:12s} | "
              f"Mood: {state.apparent_mood:12s} | "
              f"Memories: {len(state.recent_memories):3d} | "
              f"Beliefs: {len(state.beliefs):3d}",
              end="", flush=True)

    def _save_checkpoint(self):
        """Save a simulation checkpoint."""
        if self.mind:
            checkpoint_path = f"models/checkpoint_step_{self.step_count}.pt"
            Path("models").mkdir(exist_ok=True)
            self.mind.save_state(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    def stop(self):
        """Stop the simulation."""
        self.running = False
        print("\n\n💾 Saving final state...")

        if self.mind:
            final_path = "models/final_state.pt"
            Path("models").mkdir(exist_ok=True)
            self.mind.save_state(final_path)
            print(f"✅ State saved to {final_path}")

        print(f"\n📊 Simulation Statistics:")
        print(f"   Total steps: {self.step_count}")
        if self.mind:
            print(f"   Final stage: {self.mind.current_stage.name}")
            print(f"   Memories: {len(self.mind.long_term_memory)}")
            print(f"   Beliefs: {len(self.mind.belief_network.beliefs)}")

        print("\n👋 Goodbye!\n")


def cmd_run(args):
    """Run the simulation.

    Args:
        args: Command-line arguments
    """
    runner = SimulationRunner(config_path=args.config)
    runner.run(max_steps=args.steps, save_interval=args.save_interval)


def cmd_dashboard(args):
    """Launch the interactive dashboard.

    Args:
        args: Command-line arguments
    """
    print("🌐 Launching NeuralChild Dashboard...")
    print(f"📍 Config: {args.config}")
    print(f"🔌 Port: {args.port}")
    print("\nStarting Dash server...\n")

    try:
        from neuralchild.dashboard import create_dashboard
        app = create_dashboard(config_path=args.config)
        app.run_server(debug=args.debug, port=args.port, host=args.host)
    except ImportError:
        print("❌ Dashboard not yet implemented. Coming soon!")
        print("   Install with: pip install dash dash-bootstrap-components")
        sys.exit(1)


def cmd_init(args):
    """Initialize a new configuration file.

    Args:
        args: Command-line arguments
    """
    config_path = Path(args.output)

    if config_path.exists() and not args.force:
        print(f"❌ Configuration file already exists: {config_path}")
        print("   Use --force to overwrite")
        sys.exit(1)

    print(f"📝 Creating configuration file: {config_path}")

    # Create default config
    from neuralchild import Config
    config = Config()
    config.to_yaml(str(config_path))

    print(f"✅ Configuration file created!")
    print(f"\nEdit {config_path} to customize your simulation.")
    print("\nKey settings:")
    print(f"  - Developmental stage: {config.mind.starting_stage}")
    print(f"  - Learning rate: {config.mind.learning_rate}")
    print(f"  - LLM model: {config.model.llm_model}")
    print(f"  - Simulated mode: {config.development.simulate_llm}")


def cmd_info(args):
    """Display system information.

    Args:
        args: Command-line arguments
    """
    print("=" * 60)
    print("🧠 NeuralChild - AI Brain Simulation Framework")
    print("=" * 60)
    print(f"\nVersion: {__version__}")
    print("Organization: Celaya Solutions AI Research Lab")
    print("License: MIT")
    print("\n📦 Package Information:")
    print(f"   Name: neuralchild")
    print(f"   Version: {__version__}")

    # Check dependencies
    print("\n🔧 Dependencies:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch: Not installed")

    try:
        import pydantic
        print(f"   ✅ Pydantic: {pydantic.__version__}")
    except ImportError:
        print("   ❌ Pydantic: Not installed")

    try:
        import dash
        print(f"   ✅ Dash: {dash.__version__}")
    except ImportError:
        print("   ⚠️  Dash: Not installed (optional for dashboard)")

    try:
        import openai
        print(f"   ✅ OpenAI: {openai.__version__}")
    except ImportError:
        print("   ⚠️  OpenAI: Not installed (optional, can use simulated mode)")

    # System info
    print("\n💻 System:")
    import platform
    print(f"   Python: {platform.python_version()}")
    print(f"   Platform: {platform.system()} {platform.release()}")

    # Config info
    if args.config:
        print(f"\n⚙️  Configuration: {args.config}")
        try:
            config = load_config(args.config)
            print(f"   Simulated LLM: {config.development.simulate_llm}")
            print(f"   Starting stage: {config.mind.starting_stage}")
            print(f"   Debug mode: {config.development.debug_mode}")
        except Exception as e:
            print(f"   ⚠️  Could not load config: {e}")

    print("\n📚 Documentation:")
    print("   GitHub: https://github.com/celayasolutions/neuralchild")
    print("   Docs: https://neuralchild.readthedocs.io")
    print("\n" + "=" * 60)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="NeuralChild - AI Brain Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralchild run                    # Run simulation with default config
  neuralchild run --steps 1000       # Run for 1000 steps
  neuralchild dashboard              # Launch interactive dashboard
  neuralchild init                   # Create default config file
  neuralchild info                   # Show system information

For more information, visit: https://github.com/celayasolutions/neuralchild
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"NeuralChild {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the simulation")
    run_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    run_parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Maximum number of steps (default: infinite)"
    )
    run_parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)"
    )
    run_parser.set_defaults(func=cmd_run)

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch interactive dashboard")
    dashboard_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    dashboard_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8050,
        help="Port to run dashboard on (default: 8050)"
    )
    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    dashboard_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--output", "-o",
        default="config.yaml",
        help="Output path for config file (default: config.yaml)"
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing configuration"
    )
    init_parser.set_defaults(func=cmd_init)

    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    info_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    info_parser.set_defaults(func=cmd_info)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.exception("Command failed")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
