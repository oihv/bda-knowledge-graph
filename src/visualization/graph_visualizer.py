"""
Graph Visualization Module
Create interactive visualizations of the knowledge graph
"""
import ast
import json
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
from typing import Dict, List, Optional
import logging
from pathlib import Path
from src.graph_builder.neo4j_client import Neo4jClient
from config import VIS_HEIGHT, VIS_WIDTH, VISUALIZATION_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize knowledge graph using various methods"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize visualizer
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self.client = neo4j_client
        self.color_map = {
            'Company': '#FF6B6B',
            'Person': '#4ECDC4',
            'Product': '#45B7D1',
            'Investment': '#FFA07A',
            'Subsidiary': '#98D8C8',
            'Technology': '#F7DC6F',
            'Location': '#BB8FCE',
            'Event': '#85C1E2',
            'FinancialMetric': '#F8B739'
        }

    # ------------------------- UTILITY PARSER -------------------------

    def _safe_parse(self, node_data):
        """
        Safely convert Neo4j node or serialized object to dict.

        Handles dicts, Neo4j Node objects, stringified dicts, or primitives.
        """
        if not node_data:
            return {}
        # Case 1: Already dict
        if isinstance(node_data, dict):
            return node_data
        # Case 2: Neo4j Node or Relationship (has .items())
        if hasattr(node_data, "items"):
            return dict(node_data.items())
        # Case 3: Stringified node (e.g., "{'name': 'Apple', 'type': 'Company'}")
        if isinstance(node_data, str):
            try:
                parsed = ast.literal_eval(node_data)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"name": str(node_data)}
        # Case 4: Fallback
        return {"name": str(node_data)}

    # ------------------------- FETCH GRAPH DATA -------------------------

    def fetch_graph_data(self, limit: int = 200) -> Dict:
        """
        Fetch graph data from Neo4j
        
        Args:
            limit: Maximum number of nodes to fetch
            
        Returns:
            Dictionary with nodes and edges
        """
        query = f"""
        MATCH (n)
        WITH n LIMIT {limit}
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        
        results = self.client.execute_cypher(query)
        nodes, edges = {}, []

        for record in results:
            n = self._safe_parse(record.get('n'))
            m = self._safe_parse(record.get('m'))
            r = record.get('r')

            # Add source node
            if n.get('name'):
                nid = n['name']
                if nid not in nodes:
                    nodes[nid] = {
                        'id': nid,
                        'label': n.get('name', ''),
                        'properties': n
                    }

            # Add target node
            if m.get('name'):
                mid = m['name']
                if mid not in nodes:
                    nodes[mid] = {
                        'id': mid,
                        'label': m.get('name', ''),
                        'properties': m
                    }

            # Add relationship
            if r:
                rel_type = getattr(r, "type", None) or getattr(r, "r", None) or "RELATED"
                src = n.get('name')
                tgt = m.get('name')
                if src and tgt:
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'type': rel_type
                    })
        
        return {'nodes': list(nodes.values()), 'edges': edges}
    
    # ------------------------- PYVIS NETWORK -------------------------

    def create_pyvis_network(self, data: Dict = None, 
                            output_file: str = "knowledge_graph.html") -> str:
        """
        Create interactive HTML visualization using PyVis
        
        Args:
            data: Graph data (if None, fetches from Neo4j)
            output_file: Output HTML filename
            
        Returns:
            Path to generated HTML file
        """
        if data is None:
            logger.info("Fetching graph data from Neo4j")
            data = self.fetch_graph_data()
        
        logger.info(f"Creating PyVis visualization with {len(data['nodes'])} nodes")

        # Create network
        net = Network(
            height=VIS_HEIGHT,
            width=VIS_WIDTH,
            bgcolor='#222222',
            font_color='white',
            notebook=False
        )
        
        # Configure physics
        net.set_options("""
        {
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -30000,
            "centralGravity": 0.3,
            "springLength": 150,
            "springConstant": 0.04,
            "damping": 0.09
            },
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
        }
        """)

        # ---------------- NEW NODE SECTION ----------------
        for node in data['nodes']:
            node_type = node['properties'].get('type', 'Unknown')
            color = self.color_map.get(node_type, '#95A5A6')

            title = f"<b>{node['label']}</b><br>"
            title += f"Type: {node_type}<br>"
            
            # NEW: Show original names if present
            if 'original_names' in node['properties']:
                orig_names = node['properties']['original_names']
                if orig_names and len(orig_names) > 1:
                    title += f"Also known as: {', '.join(orig_names[:3])}<br>"
            
            # NEW: Show source document if present
            if 'source_document' in node['properties']:
                title += f"Source: {node['properties']['source_document']}<br>"
            
            # Add remaining properties (excluding redundant ones)
            for k, v in node['properties'].items():
                if k not in ('name', 'type', 'original_names', 'source_document', 'source_type'):
                    # Limit long values for readability
                    v_str = str(v)
                    if len(v_str) > 50:
                        v_str = v_str[:47] + '...'
                    title += f"{k}: {v_str}<br>"

            net.add_node(
                node['id'],
                label=node['label'],
                title=title,
                color=color,
                size=25
            )

        # Add edges
        for edge in data['edges']:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge['type'],
                label=edge['type']
            )
        
        output_path = Path(VISUALIZATION_DIR) / output_file
        net.save_graph(str(output_path))
        logger.info(f"Visualization saved to {output_path}")
        return str(output_path)

    # ------------------------- PLOTLY NETWORK -------------------------

    def create_plotly_network(self, data: Dict = None,
                              output_file: str = "plotly_graph.html") -> str:
        """
        Create static/interactive visualization using Plotly
        
        Args:
            data: Graph data (if None, fetches from Neo4j)
            output_file: Output HTML filename
            
        Returns:
            Path to generated HTML file
        """
        if data is None:
            logger.info("Fetching graph data from Neo4j")
            data = self.fetch_graph_data()
        
        logger.info(f"Creating Plotly visualization with {len(data['nodes'])} nodes")
        
        G = nx.Graph()
        for node in data['nodes']:
            G.add_node(node['id'], **node['properties'])
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], type=edge['type'])
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        edge_x, edge_y = [], []

        for edge in data['edges']:
            if edge['source'] in pos and edge['target'] in pos:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in data['nodes']:
            if node['id'] not in pos:
                continue
            x, y = pos[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_type = node['properties'].get('type', 'Unknown')
            node_text.append(f"{node['label']}<br>Type: {node_type}")
            node_color.append(self.color_map.get(node_type, '#95A5A6'))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n['label'] for n in data['nodes']],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=15,
                line_width=2
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Financial Knowledge Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white')
            )
        )
        
        output_path = Path(VISUALIZATION_DIR) / output_file
        fig.write_html(str(output_path))
        logger.info(f"Plotly visualization saved to {output_path}")
        return str(output_path)

    # ------------------------- SUBGRAPH VISUALIZATION -------------------------

    def create_subgraph_visualization(self, center_node: str, 
                                      depth: int = 2,
                                      output_file: str = "subgraph.html") -> Optional[str]:
        """
        Visualize subgraph around a specific node
        """
        logger.info(f"Creating subgraph visualization for {center_node} (depth={depth})")
        
        query = f"""
        MATCH path = (center {{name: $center_node}})-[*1..{depth}]-(connected)
        WITH center, connected, relationships(path) as rels
        RETURN center, connected, rels
        """
        
        results = self.client.execute_cypher(query, {'center_node': center_node})
        if not results:
            logger.warning(f"No data found for node: {center_node}")
            return None
        
        nodes, edges = {}, []
        for record in results:
            c = self._safe_parse(record.get('center'))
            conn = self._safe_parse(record.get('connected'))
            rels = record.get('rels', [])
            
            if c.get('name') and c['name'] not in nodes:
                nodes[c['name']] = {'id': c['name'], 'label': c['name'], 'properties': c}
            if conn.get('name') and conn['name'] not in nodes:
                nodes[conn['name']] = {'id': conn['name'], 'label': conn['name'], 'properties': conn}

            for rel in rels:
                src = getattr(rel, "start_node", None)
                tgt = getattr(rel, "end_node", None)
                typ = getattr(rel, "type", "RELATED")
                if src and tgt:
                    sname = getattr(src, "get", lambda k: src.get(k, None))("name") if isinstance(src, dict) else getattr(src, "get", lambda k: None)("name")
                    tname = getattr(tgt, "get", lambda k: tgt.get(k, None))("name") if isinstance(tgt, dict) else getattr(tgt, "get", lambda k: None)("name")
                    if sname and tname:
                        edges.append({'source': sname, 'target': tname, 'type': typ})
        
        data = {'nodes': list(nodes.values()), 'edges': edges}
        return self.create_pyvis_network(data, output_file)

    # ------------------------- EXPORT GRAPH -------------------------

    def export_graph_data(self, output_file: str = "graph_data.json") -> str:
        """Export full graph to JSON"""
        data = self.fetch_graph_data(limit=1000)
        output_path = Path(VISUALIZATION_DIR) / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Graph data exported to {output_path}")
        return str(output_path)

    # ------------------------- STATISTICS VISUALIZATION -------------------------

    def create_statistics_visualization(self, output_file: str = "statistics.html") -> str:
        """Create visualization of graph statistics"""
        stats = self.client.get_graph_statistics()
        node_types = stats.get('node_types', {})
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(node_types.keys()),
            y=list(node_types.values()),
            marker_color='lightblue',
            name='Node Types'
        ))
        fig.update_layout(
            title='Knowledge Graph Statistics',
            xaxis_title='Node Type',
            yaxis_title='Count',
            plot_bgcolor='white'
        )
        output_path = Path(VISUALIZATION_DIR) / output_file
        fig.write_html(str(output_path))
        logger.info(f"Statistics visualization saved to {output_path}")
        return str(output_path)
