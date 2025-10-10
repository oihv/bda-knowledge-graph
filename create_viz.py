"""
Create visualizations from existing Neo4j graph
Run this after successful graph construction
"""
from src.graph_builder.neo4j_client import Neo4jClient
from src.visualization.graph_visualizer import GraphVisualizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n🎨 Creating Graph Visualizations\n")
    
    try:
        # Connect to Neo4j
        client = Neo4jClient()
        client.connect()
        
        # Get statistics
        stats = client.get_graph_statistics()
        print(f"📊 Graph Statistics:")
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Relationships: {stats['total_relationships']}")
        print(f"\n   Node Types:")
        for node_type, count in stats['node_types'].items():
            print(f"      {node_type}: {count}")
        
        # Create visualizer
        visualizer = GraphVisualizer(client)
        
        print("\n🔨 Creating visualizations...")
        
        # 1. PyVis Interactive Network
        print("   1. Interactive PyVis network...", end=" ")
        try:
            pyvis_file = visualizer.create_pyvis_network("financial_graph.html")
            print(f"✅\n      → {pyvis_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # 2. Plotly Graph
        print("   2. Plotly visualization...", end=" ")
        try:
            plotly_file = visualizer.create_plotly_network("financial_plotly.html")
            print(f"✅\n      → {plotly_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # 3. Statistics
        print("   3. Statistics dashboard...", end=" ")
        try:
            stats_file = visualizer.create_statistics_visualization("financial_stats.html")
            print(f"✅\n      → {stats_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # 4. Export JSON
        print("   4. JSON data export...", end=" ")
        try:
            json_file = visualizer.export_graph_data("financial_graph_data.json")
            print(f"✅\n      → {json_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        client.close()
        
        print("\n✅ Visualization creation complete!")
        print("\nNext steps:")
        print("   1. Open HTML files in your browser")
        print("   2. Or view in Neo4j Browser: http://localhost:7474")
        print("   3. Run: MATCH (n) RETURN n LIMIT 100")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure Neo4j is running")
        print("   - Check your connection in config.py")

if __name__ == "__main__":
    main()