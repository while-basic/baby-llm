#----------------------------------------------------------------------------
#File:       __init__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Package initializer for the visualization module.
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Package initializer for the visualization module."""

# Import visualization components
try:
    from neural_child.visualization.visualization import (
        EmotionalStateVisualizer,
        NeuralNetworkVisualizer,
        plot_learning_metrics,
        plot_psychological_metrics
    )
except ImportError:
    EmotionalStateVisualizer = None
    NeuralNetworkVisualizer = None
    plot_learning_metrics = None
    plot_psychological_metrics = None
    print("Warning: Visualization components not available.")

__all__ = [
    'EmotionalStateVisualizer',
    'NeuralNetworkVisualizer',
    'plot_learning_metrics',
    'plot_psychological_metrics'
]

