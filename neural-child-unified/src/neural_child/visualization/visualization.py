#----------------------------------------------------------------------------
#File:       visualization.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Module for visualizing emotional states and neural network architectures
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Module for visualizing emotional states and neural network architectures.

Extracted from neural-child-init/visualization.py
Adapted imports to use unified structure.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

# Optional imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Warning: Matplotlib not available. Plotting features will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None
    print("Warning: Seaborn not available. Some visualization features will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("Warning: PyTorch not available. Tensor operations will be limited.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    print("Warning: NetworkX not available. Network visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    print("Warning: Plotly not available. 3D visualizations will be limited.")

# Set style if available
if SEABORN_AVAILABLE:
    sns.set_style("whitegrid")
if MATPLOTLIB_AVAILABLE:
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('dark_background')


class EmotionalStateVisualizer:
    """Visualizer for emotional states over time."""
    
    def __init__(self):
        """Initialize the emotional state visualizer."""
        self.emotional_history: List[np.ndarray] = []
        self.time_points: List[Any] = []
    
    def add_emotional_state(self, emotional_state: Any, timestamp: Any):
        """Add an emotional state to the history.
        
        Args:
            emotional_state: Emotional state (tensor, array, or dict)
            timestamp: Timestamp for this state
        """
        if TORCH_AVAILABLE and isinstance(emotional_state, torch.Tensor):
            emotional_state = emotional_state.cpu().detach().numpy()
        elif isinstance(emotional_state, dict):
            # Convert dict to array (assuming 4 basic emotions)
            emotional_state = np.array([
                emotional_state.get('joy', 0.0),
                emotional_state.get('trust', 0.0),
                emotional_state.get('fear', 0.0),
                emotional_state.get('surprise', 0.0)
            ])
        elif not isinstance(emotional_state, np.ndarray):
            emotional_state = np.array(emotional_state)
            
        self.emotional_history.append(emotional_state)
        self.time_points.append(timestamp)
    
    def get_emotional_data(self) -> Dict[str, Any]:
        """Get emotional data as dictionary for JSON serialization.
        
        Returns:
            Dictionary with emotional data for visualization
        """
        if not self.emotional_history:
            return {
                'emotions': ['Joy', 'Trust', 'Fear', 'Surprise'],
                'data': [],
                'time_points': []
            }
        
        data = np.array(self.emotional_history)
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        return {
            'emotions': emotions,
            'data': data.tolist(),
            'time_points': [str(t) for t in self.time_points]
        }
    
    def plot_emotional_timeline(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Plot emotional state timeline.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Figure object if matplotlib available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create plot.")
            return None
            
        if not self.emotional_history:
            print("No emotional data to plot.")
            return None
        
        data = np.array(self.emotional_history)
        time_points = np.array(self.time_points)
        
        plt.figure(figsize=(15, 8))
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        for i, emotion in enumerate(emotions):
            if i < data.shape[1]:
                plt.plot(time_points, data[:, i], label=emotion, linewidth=2)
        
        plt.title('Emotional State Timeline', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Emotional Intensity', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            return plt.gcf()
    
    def create_emotional_heatmap(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Create emotional state correlation heatmap.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Figure object if matplotlib/seaborn available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            print("Matplotlib or Seaborn not available. Cannot create heatmap.")
            return None
            
        if not self.emotional_history:
            print("No emotional data to plot.")
            return None
        
        data = np.array(self.emotional_history)
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        if data.shape[1] < len(emotions):
            # Pad data if needed
            padded_data = np.zeros((data.shape[0], len(emotions)))
            padded_data[:, :data.shape[1]] = data
            data = padded_data
        
        correlations = np.corrcoef(data.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, 
                   annot=True, 
                   cmap='coolwarm', 
                   xticklabels=emotions,
                   yticklabels=emotions)
        plt.title('Emotional State Correlations', fontsize=16)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            return plt.gcf()
    
    def create_3d_emotional_trajectory(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Create 3D emotional state trajectory.
        
        Args:
            save_path: Optional path to save the plot (HTML format)
            
        Returns:
            Plotly figure if available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create 3D trajectory.")
            return None
            
        if not self.emotional_history:
            print("No emotional data to plot.")
            return None
        
        data = np.array(self.emotional_history)
        
        if data.shape[1] < 4:
            # Pad data if needed
            padded_data = np.zeros((data.shape[0], 4))
            padded_data[:, :data.shape[1]] = data
            data = padded_data
        
        fig = go.Figure(data=[go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=data[:, 3],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Surprise Level')
            ),
            line=dict(color='darkblue', width=2),
            text=[f'Time: {t}<br>Joy: {d[0]:.2f}<br>Trust: {d[1]:.2f}<br>Fear: {d[2]:.2f}<br>Surprise: {d[3]:.2f}'
                  for t, d in zip(self.time_points, data)],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title='3D Emotional State Trajectory',
            scene=dict(
                xaxis_title='Joy',
                yaxis_title='Trust',
                zaxis_title='Fear'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig


class NeuralNetworkVisualizer:
    """Visualizer for neural network architectures."""
    
    def __init__(self, model: Optional[Any] = None):
        """Initialize the neural network visualizer.
        
        Args:
            model: PyTorch model to visualize (optional)
        """
        self.model = model
    
    def get_architecture_data(self) -> Dict[str, Any]:
        """Get network architecture data as dictionary for JSON serialization.
        
        Returns:
            Dictionary with architecture data
        """
        if not self.model or not TORCH_AVAILABLE:
            return {'layers': [], 'connections': []}
        
        layers = []
        connections = []
        
        layer_idx = 0
        prev_layer_idx = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append({
                    'name': name,
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features
                })
                
                if prev_layer_idx is not None:
                    connections.append({
                        'from': prev_layer_idx,
                        'to': layer_idx
                    })
                
                prev_layer_idx = layer_idx
                layer_idx += 1
        
        return {
            'layers': layers,
            'connections': connections
        }
    
    def visualize_architecture(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize network architecture using matplotlib/networkx.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Figure object if available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE or not self.model:
            print("Matplotlib, NetworkX, or model not available. Cannot visualize architecture.")
            return None
        
        G = nx.DiGraph()
        
        # Add nodes for each layer
        layer_sizes = []
        layer_names = []
        
        # Extract layer information
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_sizes.append(module.in_features)
                layer_names.append(f"{name}\n({module.in_features}→{module.out_features})")
                if len(layer_sizes) == 1:  # First layer
                    G.add_node(0, name=layer_names[-1], size=module.in_features)
                G.add_node(len(layer_sizes), name=f"{name}_out\n({module.out_features})", size=module.out_features)
                G.add_edge(len(layer_sizes)-1, len(layer_sizes))
        
        if not G.nodes():
            print("No layers found to visualize.")
            return None
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_sizes = [1000 + G.nodes[node].get('size', 100)*10 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=node_sizes,
                             alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: G.nodes[node].get('name', str(node)) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title("Neural Network Architecture", pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
            plt.close()
            return None
        else:
            return plt.gcf()


def plot_learning_metrics(telemetry: Dict[str, Any], save_path: Optional[str] = None) -> Optional[Any]:
    """Plot learning metrics from telemetry data.
    
    Args:
        telemetry: Dictionary with learning metrics
        save_path: Optional path to save the plot
        
    Returns:
        Figure object if matplotlib available, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return None
    
    metrics = ['loss', 'emotional_stability', 'conversation_quality']
    available_metrics = [m for m in metrics if m in telemetry and telemetry[m]]
    
    if not available_metrics:
        print("No learning metrics found in telemetry.")
        return None
    
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(15, 5*len(available_metrics)))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        axes[i].plot(telemetry[metric], label=metric)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        return fig


def plot_psychological_metrics(telemetry: Dict[str, Any], save_path: Optional[str] = None) -> Optional[Any]:
    """Plot psychological development metrics.
    
    Args:
        telemetry: Dictionary with psychological metrics
        save_path: Optional path to save the plot (HTML format)
        
    Returns:
        Plotly figure if available, None otherwise
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create psychological metrics plot.")
        return None
    
    if 'psychological' not in telemetry:
        print("No psychological metrics found in telemetry.")
        return None
    
    metrics = telemetry['psychological']
    categories = list(metrics.keys())
    values = [np.mean(metrics[cat]) if metrics[cat] else 0 for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Psychological Development'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Psychological Development Metrics'
    )
    
    if save_path:
        fig.write_html(save_path)
    return fig

