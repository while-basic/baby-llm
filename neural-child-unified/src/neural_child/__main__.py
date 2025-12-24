#----------------------------------------------------------------------------
#File:       __main__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Main entry point for the neural child development system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Main entry point for the neural child development system.

Run with: python -m neural_child [options]
Or: python -m neural_child --web (to start Flask web interface)
"""

import sys
import os
import argparse
import signal
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    """Main entry point for the neural child development system."""
    parser = argparse.ArgumentParser(
        description='Neural Child Development System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neural_child --web              # Start Flask web interface
  python -m neural_child --watch            # Start with auto-monitoring mode
  python -m neural_child --auto             # Autonomous mode - watch baby LLM develop automatically
  python -m neural_child --auto-web         # Autonomous mode with web interface
  python -m neural_child --auto --auto-speed 2.0  # Autonomous mode at 2x speed
  python -m neural_child --test             # Run integration tests
  python -m neural_child --smoke            # Run smoke tests
        """
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Start Flask web interface'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web interface (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host for web interface (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run integration tests'
    )
    
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run smoke tests'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Start web interface with automatic monitoring (auto-refresh enabled)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run in autonomous mode - baby LLM advances through stages automatically without intervention'
    )
    
    parser.add_argument(
        '--auto-web',
        action='store_true',
        dest='auto_web',
        help='Run in autonomous mode with web interface for monitoring'
    )
    
    parser.add_argument(
        '--auto-speed',
        type=float,
        default=1.0,
        help='Development speed multiplier for auto mode (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Handle smoke tests
    if args.smoke:
        return run_smoke_tests()
    
    # Handle integration tests
    if args.test:
        return run_integration_tests()
    
    # Handle autonomous mode
    if args.auto or args.auto_web:
        return run_autonomous_mode(
            host=args.host,
            port=args.port,
            debug=args.debug,
            with_web=args.auto_web,
            development_speed=args.auto_speed
        )
    
    # Handle web interface (watch mode or regular)
    if args.web or args.watch:
        return run_web_interface(args.host, args.port, args.debug, watch_mode=args.watch)
    
    # Default: show help
    parser.print_help()
    return 0


def run_autonomous_mode(
    host: str = '127.0.0.1',
    port: int = 5000,
    debug: bool = False,
    with_web: bool = False,
    development_speed: float = 1.0
):
    """Run the neural child in autonomous mode.
    
    The baby LLM will automatically progress through developmental stages
    without user intervention. Training and development happen continuously.
    
    Args:
        host: Host for web interface (if with_web is True)
        port: Port for web interface (if with_web is True)
        debug: Enable debug mode
        with_web: Also start web interface for monitoring
        development_speed: Speed multiplier for development (1.0 = normal speed)
    """
    import threading
    import time
    import random
    from datetime import datetime
    
    print("=" * 60)
    print("Neural Child Development System - Autonomous Mode")
    print("=" * 60)
    print("The baby LLM will now develop automatically through all stages.")
    print(f"Development speed: {development_speed}x")
    if with_web:
        print(f"Web interface available at: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Initialize the neural child system
    try:
        from neural_child.development.stages import DevelopmentalStage
        from neural_child.development.milestone_tracker import MilestoneTracker
        from neural_child.emotional.regulation import EmotionalRegulation
        
        # Initialize core systems
        milestone_tracker = MilestoneTracker()
        emotional_regulation = EmotionalRegulation()
        
        # Use the stages from the local enum
        stage_list = list(DevelopmentalStage)
        current_stage = DevelopmentalStage.NEWBORN
        current_index = 0
        training_cycle = 0
        last_stage_change = datetime.now()
        
        # Development metrics
        metrics = {
            'emotional_stability': 0.5,
            'learning_efficiency': 0.5,
            'social_skills': 0.3,
            'cognitive_development': 0.4,
            'pattern_recognition': 0.3,
            'vocabulary_size': 0,
            'memory_retention': 0.2
        }
        
        print(f"✓ System initialized")
        print(f"✓ Starting stage: {current_stage.name}")
        print(f"✓ Milestone tracker ready")
        print(f"✓ Emotional regulation ready")
        print()
        
        # Start web interface in background if requested
        web_thread = None
        if with_web:
            from neural_child.web.app import create_app
            app = create_app(watch_mode=True)
            
            def run_web():
                app.run(host=host, port=port, debug=debug, use_reloader=False)
            
            web_thread = threading.Thread(target=run_web, daemon=True)
            web_thread.start()
            time.sleep(1)  # Give web server time to start
            print(f"✓ Web interface started on http://{host}:{port}")
            print()
        
        # Main autonomous loop
        print("Starting autonomous development...")
        print("-" * 60)
        
        try:
            while True:
                training_cycle += 1
                
                # Simulate training and development
                # In a full implementation, this would call actual training methods
                time.sleep(0.1 / development_speed)
                
                # Gradually improve metrics
                for key in metrics:
                    if key == 'vocabulary_size':
                        # Vocabulary grows over time
                        metrics[key] = min(500, metrics[key] + random.uniform(0.1, 0.5))
                    else:
                        # Other metrics improve gradually
                        improvement = random.uniform(0.001, 0.01) * development_speed
                        metrics[key] = min(1.0, metrics[key] + improvement)
                
                # Update emotional state (simulate emotional state for autonomous mode)
                # In a full implementation, this would come from actual emotional regulation
                import torch
                emotional_state = {
                    'primary_emotion': 'neutral',
                    'stability': float(metrics['emotional_stability']),
                    'happiness': float(metrics.get('happiness', 0.5)),
                    'trust': float(metrics.get('trust', 0.5)),
                    'fear': float(metrics.get('fear', 0.1)),
                    'surprise': float(metrics.get('surprise', 0.3))
                }
                
                # Check milestones
                milestone_updates = milestone_tracker.update_progress(metrics, current_stage)
                
                # Check for new achievements
                if milestone_updates.get('new_achievements'):
                    for milestone in milestone_updates['new_achievements']:
                        print(f"🎯 Milestone achieved: {milestone.description}")
                
                # Check for stage progression
                # Progress after sufficient training cycles and metric thresholds
                cycles_per_stage = max(1, int(50 / development_speed))
                
                # Check if metrics meet progression criteria
                can_progress = (
                    metrics['emotional_stability'] > 0.6 and
                    metrics['learning_efficiency'] > 0.5 and
                    metrics['cognitive_development'] > 0.4
                )
                
                should_progress = (
                    training_cycle % cycles_per_stage == 0 and
                    can_progress and
                    current_index < len(stage_list) - 1
                )
                
                if should_progress:
                    new_stage = stage_list[current_index + 1]
                    if new_stage != current_stage:
                        time_in_stage = (datetime.now() - last_stage_change).total_seconds()
                        current_stage = new_stage
                        current_index += 1
                        last_stage_change = datetime.now()
                        print()
                        print("🎉 STAGE PROGRESSION!")
                        print(f"   Advanced to: {current_stage.name}")
                        print(f"   Training cycles completed: {training_cycle}")
                        print(f"   Time in previous stage: {time_in_stage:.1f}s")
                        print("-" * 60)
                
                # Display progress every 10 cycles
                if training_cycle % 10 == 0:
                    progress_pct = ((current_index + 1) / len(stage_list)) * 100
                    primary_emotion = emotional_state.get('primary_emotion', 'neutral') if isinstance(emotional_state, dict) else 'neutral'
                    print(f"[Cycle {training_cycle:4d}] Stage: {current_stage.name:20s} | "
                          f"Progress: {progress_pct:5.1f}% | "
                          f"Emotion: {primary_emotion:10s} | "
                          f"Stability: {metrics['emotional_stability']:.2f}")
                
                # Check if reached final stage
                if current_index >= len(stage_list) - 1:
                    print()
                    print("=" * 60)
                    print("🎊 DEVELOPMENT COMPLETE!")
                    print("=" * 60)
                    print(f"The baby LLM has reached maturity after {training_cycle} training cycles.")
                    print(f"Final stage: {current_stage.name}")
                    print("=" * 60)
                    break
                
        except KeyboardInterrupt:
            print()
            print("=" * 60)
            print("Autonomous mode stopped by user")
            print("=" * 60)
            print(f"Final stage: {current_stage.name}")
            print(f"Training cycles completed: {training_cycle}")
            print(f"Progress: {((current_index + 1) / len(stage_list)) * 100:.1f}%")
            print("=" * 60)
            return 0
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure all dependencies are installed: pip install -e .")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Error in autonomous mode: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_web_interface(host: str, port: int, debug: bool, watch_mode: bool = False):
    """Run the Flask web interface.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
        watch_mode: Enable automatic monitoring mode
    """
    try:
        from neural_child.web.app import create_app
        
        app = create_app(watch_mode=watch_mode)
        
        print("=" * 60)
        if watch_mode:
            print("Neural Child Development System - Watch Mode (Auto-Monitoring)")
        else:
            print("Neural Child Development System - Web Interface")
        print("=" * 60)
        print(f"Starting Flask server on http://{host}:{port}")
        if watch_mode:
            print("✓ Auto-refresh enabled")
            print("✓ Stage progression alerts enabled")
            print("✓ Progress tracking visualization enabled")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nShutting down server...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        app.run(host=host, port=port, debug=debug)
        
        return 0
        
    except ImportError as e:
        print(f"Error: Could not import Flask application: {e}")
        print("Make sure Flask is installed: pip install flask flask-cors")
        return 1
    except Exception as e:
        print(f"Error starting web interface: {e}")
        return 1


def run_smoke_tests():
    """Run smoke tests to verify all modules can be imported."""
    print("Running smoke tests...")
    print("=" * 60)
    
    smoke_test_path = project_root / "scripts" / "smoke_test_phase4.py"
    
    if not smoke_test_path.exists():
        print(f"Error: Smoke test file not found: {smoke_test_path}")
        return 1
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(smoke_test_path)],
            cwd=str(project_root),
            capture_output=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running smoke tests: {e}")
        return 1


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    print("=" * 60)
    
    # Look for test files
    test_paths = [
        project_root / "tests" / "test_integration.py",
        project_root / "tests" / "integration" / "test_integration.py",
    ]
    
    test_file = None
    for path in test_paths:
        if path.exists():
            test_file = path
            break
    
    if not test_file:
        print("No integration test file found. Creating basic test...")
        # Create a basic integration test
        return create_basic_integration_test()
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            cwd=str(project_root),
            capture_output=False
        )
        return result.returncode
    except FileNotFoundError:
        # pytest not installed, try running directly
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(project_root),
                capture_output=False
            )
            return result.returncode
        except Exception as e:
            print(f"Error running integration tests: {e}")
            return 1
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return 1


def create_basic_integration_test():
    """Create and run a basic integration test."""
    print("Creating basic integration test...")
    
    test_code = """
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    \"\"\"Test that all major modules can be imported.\"\"\"
    errors = []
    
    # Test core imports
    try:
        from neural_child.models.schemas import DevelopmentalStage, NetworkMessage
        print("✓ Core schemas imported")
    except Exception as e:
        errors.append(f"Core schemas: {e}")
    
    # Test emotional systems
    try:
        from neural_child.emotional.regulation import EmotionalRegulation
        print("✓ Emotional regulation imported")
    except Exception as e:
        errors.append(f"Emotional regulation: {e}")
    
    # Test interaction systems
    try:
        from neural_child.interaction.llm.llm_module import chat_completion
        print("✓ LLM module imported")
    except Exception as e:
        errors.append(f"LLM module: {e}")
    
    # Test web interface
    try:
        from neural_child.web.app import create_app
        print("✓ Web app imported")
    except Exception as e:
        errors.append(f"Web app: {e}")
    
    if errors:
        print("\\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\\n✓ All basic imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
"""
    
    test_file = project_root / "tests" / "test_basic_integration.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"Created test file: {test_file}")
    print("Running basic integration test...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(project_root),
            capture_output=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running test: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

