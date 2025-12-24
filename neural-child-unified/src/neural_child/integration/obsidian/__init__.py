#----------------------------------------------------------------------------
#File:       __init__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Package initializer for the Obsidian integration module.
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Package initializer for the Obsidian integration module."""

# Import Obsidian components
try:
    from neural_child.integration.obsidian.obsidian_api import ObsidianAPI
except ImportError:
    ObsidianAPI = None
    print("Warning: ObsidianAPI not available.")

try:
    from neural_child.integration.obsidian.obsidian_connector import ObsidianConnector
except ImportError:
    ObsidianConnector = None
    print("Warning: ObsidianConnector not available.")

try:
    from neural_child.integration.obsidian.obsidian_visualizer import ObsidianVisualizer
except ImportError:
    ObsidianVisualizer = None
    print("Warning: ObsidianVisualizer not available.")

try:
    from neural_child.integration.obsidian.obsidian_heartbeat_logger import ObsidianHeartbeatLogger
except ImportError:
    ObsidianHeartbeatLogger = None
    print("Warning: ObsidianHeartbeatLogger not available.")

__all__ = [
    'ObsidianAPI',
    'ObsidianConnector',
    'ObsidianVisualizer',
    'ObsidianHeartbeatLogger'
]

