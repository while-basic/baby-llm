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
  python -m neural_child --watch --port 5000  # Watch mode on port 5000
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
    
    args = parser.parse_args()
    
    # Handle smoke tests
    if args.smoke:
        return run_smoke_tests()
    
    # Handle integration tests
    if args.test:
        return run_integration_tests()
    
    # Handle web interface (watch mode or regular)
    if args.web or args.watch:
        return run_web_interface(args.host, args.port, args.debug, watch_mode=args.watch)
    
    # Default: show help
    parser.print_help()
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

