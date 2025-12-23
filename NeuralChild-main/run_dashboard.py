#!/usr/bin/env python
"""
Run script for the Neural Child Dashboard.

This script provides a convenient way to start the Neural Child Dashboard
with proper environment setup and error handling.
"""

import os
import sys
import logging
import subprocess
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_dashboard")

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        "torch", "numpy", "pydantic", "dash", "dash-bootstrap-components",
        "plotly", "pandas"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing_packages
            ])
            logger.info("All required packages installed successfully.")
        except subprocess.CalledProcessError:
            logger.error("Failed to install required packages.")
            logger.info("Please install them manually using: pip install -r requirements.txt")
            sys.exit(1)

def run_dashboard():
    """Run the Neural Child Dashboard."""
    logger.info("Starting Neural Child Dashboard...")
    
    # Get the absolute path to the dashboard script
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "neural-child-dashboard.py"
    )
    
    logger.info(f"Dashboard path: {dashboard_path}")
    
    try:
        # Run the dashboard script
        subprocess.run([sys.executable, dashboard_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user.")
    
if __name__ == "__main__":
    # Check if all required packages are installed
    logger.info("Checking required packages...")
    check_dependencies()
    
    # Run the dashboard
    run_dashboard()