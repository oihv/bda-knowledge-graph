"""
Architecture and Workflow Diagrams for Financial Knowledge Graph System

This script generates architecture and workflow diagrams for inclusion in documentation or presentations.

Usage:
    python docs/architecture_diagrams.py

It will output PNG files in the docs/ directory.
"""
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def create_architecture_diagram(output_dir):
    G = nx.DiGraph()
    
    # Add nodes with positions
    pos = {
        'A': (0, 2),    # User
        'B': (1, 3),    # Preprocessing
        'C': (2, 3),    # LLM
        'D': (3, 3),    # Normalization
        'E': (4, 3),    # Graph Construction
        'F': (4, 1),    # Visualization
    }
    
    labels = {
        'A': 'User\n(VS Code, Streamlit UI)',
        'B': 'Document Preprocessing\n(PDF, Text, Chunking)',
        'C': 'LLM Entity Extraction\n(OpenRouter/HuggingFace)',
        'D': 'Entity/Relationship\nNormalization',
        'E': 'Graph Construction\n(Neo4j Database)',
        'F': 'Visualization & Query\n(PyVis, Plotly, Streamlit)'
    }
    
    # Add nodes and edges
    G.add_nodes_from(pos.keys())
    edges = [
        ('A', 'B'), ('B', 'C'), ('C', 'D'), 
        ('D', 'E'), ('E', 'F'), ('A', 'F'),
        ('F', 'E')
    ]
    G.add_edges_from(edges)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw the network
    nx.draw(G, pos, 
           labels=labels,
           node_color='lightblue',
           node_size=3000,
           font_size=8,
           font_weight='bold',
           arrows=True,
           edge_color='gray',
           arrowsize=20)
    
    # Add edge labels
    edge_labels = {
        ('A', 'F'): 'Query/Visualize',
        ('F', 'E'): 'Cypher Query'
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Save the plot
    out_path = Path(output_dir) / 'architecture_diagram.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Architecture diagram saved to {out_path}")

def create_workflow_diagram(output_dir):
    G = nx.DiGraph()
    
    # Add nodes with positions (in a vertical layout)
    pos = {
        '1': (0.5, 6),  # Upload
        '2': (0.5, 5),  # Preprocess
        '3': (0.5, 4),  # LLM
        '4': (0.5, 3),  # Normalize
        '5': (0.5, 2),  # Build Graph
        '6': (0.5, 1),  # Visualize
    }
    
    labels = {
        '1': 'Upload/Select Documents',
        '2': 'Preprocess & Chunk',
        '3': 'LLM Entity Extraction',
        '4': 'Normalize Entities/Relations',
        '5': 'Build Knowledge Graph',
        '6': 'Visualize & Query'
    }
    
    # Add nodes and edges
    G.add_nodes_from(pos.keys())
    edges = [('1','2'), ('2','3'), ('3','4'), ('4','5'), ('5','6')]
    G.add_edges_from(edges)
    
    # Create figure
    plt.figure(figsize=(8, 12))
    
    # Draw the network
    nx.draw(G, pos,
           labels=labels,
           node_color='lightgreen',
           node_size=3000,
           font_size=10,
           font_weight='bold',
           edge_color='gray',
           arrowsize=20,
           width=2)
    
    # Save the plot
    out_path = Path(output_dir) / 'workflow_diagram.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Workflow diagram saved to {out_path}")

if __name__ == '__main__':
    output_dir = os.path.dirname(__file__)
    create_architecture_diagram(output_dir)
    create_workflow_diagram(output_dir)
