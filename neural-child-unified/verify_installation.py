#----------------------------------------------------------------------------
#File:       verify_installation.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Script to verify installation and setup
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Verification script for neural-child-unified installation.

Run this script to verify that the package is properly installed
and all components can be imported.

Usage:
    python verify_installation.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11."""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print("✓ Python 3.11 detected")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor} detected (3.11 required)")
        return False

def check_package_installed():
    """Check if neural_child package is installed."""
    try:
        import neural_child
        print(f"✓ Package installed at: {neural_child.__file__}")
        return True
    except ImportError:
        print("✗ Package not installed. Run: pip install -e .")
        return False

def check_core_imports():
    """Check if core modules can be imported."""
    modules = [
        ("neural_child.models.schemas", "Schemas"),
        ("neural_child.emotional.regulation", "Emotional Regulation"),
        ("neural_child.interaction.llm.llm_module", "LLM Module"),
        ("neural_child.web.app", "Web App"),
        ("neural_child.dream.dream_system", "Dream System"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} import OK")
        except ImportError as e:
            print(f"✗ {display_name} import failed: {e}")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Check if key dependencies are installed."""
    dependencies = [
        ("torch", "PyTorch"),
        ("pydantic", "Pydantic"),
        ("flask", "Flask"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} installed")
        except ImportError:
            print(f"✗ {display_name} not installed")
            all_ok = False
    
    return all_ok

def check_config_file():
    """Check if config file exists."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if config_path.exists():
        print(f"✓ Config file found: {config_path}")
        return True
    else:
        print(f"✗ Config file not found: {config_path}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Neural Child Development System - Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Checking Python version...")
    results.append(("Python Version", check_python_version()))
    print()
    
    print("2. Checking package installation...")
    results.append(("Package Installation", check_package_installed()))
    print()
    
    print("3. Checking core imports...")
    results.append(("Core Imports", check_core_imports()))
    print()
    
    print("4. Checking dependencies...")
    results.append(("Dependencies", check_dependencies()))
    print()
    
    print("5. Checking config file...")
    results.append(("Config File", check_config_file()))
    print()
    
    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Ensure Ollama is running: ollama serve")
        print("  2. Download model: ollama pull gemma3:1b")
        print("  3. Start web interface: python -m neural_child --web")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  - Install package: pip install -e .")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - See INSTALL.md for detailed instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())

