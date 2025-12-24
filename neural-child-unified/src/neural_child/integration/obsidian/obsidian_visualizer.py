#----------------------------------------------------------------------------
#File:       obsidian_visualizer.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Visualize Obsidian notes and their connections
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Visualize Obsidian notes and their connections.

Extracted from neural-child-init/obsidian_visualizer.py
Adapted imports to use unified structure.
"""

from typing import Dict, List, Set
import re
import json
from pathlib import Path
import os

# Local imports
from neural_child.integration.obsidian.obsidian_api import ObsidianAPI

# Optional imports for visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    print("Warning: NetworkX not available. Graph visualization will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Warning: Matplotlib not available. Graph visualization will be limited.")


class ObsidianVisualizer:
    """Visualizer for Obsidian vault notes and connections."""
    
    def __init__(self, vault_path: str):
        """Initialize the visualizer with local vault path.
        
        Args:
            vault_path (str): Path to Obsidian vault
        """
        self.api = ObsidianAPI(vault_path=vault_path)
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()
        else:
            self.graph = None
            print("Warning: Graph functionality disabled due to missing NetworkX.")
        
    def extract_links(self, content: str) -> Set[str]:
        """Extract wiki-style links from content.
        
        Args:
            content (str): Markdown content
            
        Returns:
            Set[str]: Set of extracted link names
        """
        # Match both [[Link]] and [[Link|Alias]] formats
        links = re.findall(r'\[\[(.*?)(?:\|.*?)?\]\]', content)
        return set(links)
        
    def build_graph(self):
        """Build a graph of notes and their connections."""
        if not NETWORKX_AVAILABLE:
            print("Error: NetworkX not available. Cannot build graph.")
            return
            
        try:
            # Get all files in the vault
            files = self.api.list_vault_files()
            print(f"Found {len(files)} files in vault")
            
            # Process each markdown file
            for file in files:
                try:
                    # Get file content
                    content = self.api.get_vault_file(file)
                    
                    # Add node for this file
                    node_name = Path(file).stem
                    self.graph.add_node(node_name)
                    
                    # Extract and add links
                    links = self.extract_links(content)
                    for link in links:
                        self.graph.add_edge(node_name, link)
                        
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
            
            print(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"Error building graph: {str(e)}")
    
    def visualize(self, output_file: str = "obsidian_graph.png"):
        """Create a visualization of the note graph.
        
        Args:
            output_file (str): Output file path for the visualization
        """
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Error: NetworkX or Matplotlib not available. Cannot create visualization.")
            return
            
        try:
            if self.graph is None or self.graph.number_of_nodes() == 0:
                print("No nodes in graph to visualize")
                return
                
            # Create the plot
            plt.figure(figsize=(15, 10))
            
            # Calculate node sizes based on connections
            node_sizes = [3000 * (1 + self.graph.degree(node)) for node in self.graph.nodes()]
            
            # Calculate node colors based on number of connections
            node_colors = [self.graph.degree(node) for node in self.graph.nodes()]
            
            # Create the layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Draw the graph
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=0.5,
                alpha=0.7,
                cmap=plt.cm.viridis
            )
            
            # Save the visualization
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
            
            # Generate statistics
            stats = {
                'total_notes': self.graph.number_of_nodes(),
                'total_connections': self.graph.number_of_edges(),
                'most_connected': sorted(
                    [(node, self.graph.degree(node)) for node in self.graph.nodes()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if self.graph.number_of_nodes() > 0 else []
            }
            
            # Save statistics
            stats_file = Path(output_file).with_suffix('.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_file}")
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def analyze_connections(self) -> Dict:
        """Analyze the connections between notes.
        
        Returns:
            Dict: Analysis results with most connected notes, isolated notes, clusters, etc.
        """
        if not NETWORKX_AVAILABLE:
            print("Error: NetworkX not available. Cannot analyze connections.")
            return {}
            
        try:
            if self.graph is None or self.graph.number_of_nodes() == 0:
                print("No nodes in graph to analyze")
                return {}
                
            analysis = {
                'most_connected': [],
                'isolated_notes': [],
                'clusters': [],
                'avg_connections': 0
            }
            
            # Find most connected notes
            analysis['most_connected'] = sorted(
                [(node, self.graph.degree(node)) for node in self.graph.nodes()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Find isolated notes
            analysis['isolated_notes'] = [
                node for node in self.graph.nodes()
                if self.graph.degree(node) == 0
            ]
            
            # Find clusters/communities
            if self.graph.number_of_nodes() > 0:
                try:
                    communities = list(nx.community.greedy_modularity_communities(self.graph))
                    analysis['clusters'] = [
                        [node for node in community]
                        for community in communities
                    ]
                except Exception as e:
                    print(f"Warning: Could not compute communities: {str(e)}")
                    analysis['clusters'] = []
            
            # Calculate average connections
            if self.graph.number_of_nodes() > 0:
                analysis['avg_connections'] = sum(
                    self.graph.degree(node) for node in self.graph.nodes()
                ) / self.graph.number_of_nodes()
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing connections: {str(e)}")
            return {}

