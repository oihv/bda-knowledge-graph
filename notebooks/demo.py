# Financial Knowledge Graph - Demo Notebook
# This notebook demonstrates the complete pipeline interactively

# Cell 1: Setup and Imports
"""
## Financial Knowledge Graph Extraction System
### Interactive Demo Notebook

This notebook walks through the complete pipeline:
1. Document preprocessing
2. Entity extraction with LLM
3. Graph construction in Neo4j
4. Visualization
5. Natural language queries
"""

import sys
sys.path.append('..')

from config import *
from src.preprocessing.pdf_processor import create_sample_text_documents
from src.preprocessing.text_cleaner import TextCleaner
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.llm_extraction.entity_extractor import EntityExtractor
from src.graph_builder.neo4j_client import Neo4jClient
from src.graph_builder.graph_constructor import GraphConstructor
from src.visualization.graph_visualizer import GraphVisualizer
from src.query_interface.nl_to_cypher import NLToCypherConverter, QueryInterface

import json
import pandas as pd
from IPython.display import HTML, display, IFrame
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful!")

# Cell 2: Load Sample Documents
"""
### Step 1: Load Sample Financial Documents

We'll use pre-built sample documents about Samsung, Apple, and Tesla.
"""

documents = create_sample_text_documents()

print(f"📄 Loaded {len(documents)} sample documents:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc['filename']} ({len(doc['text'])} characters)")

# Preview first document
print("\n--- Preview of first document ---")
print(documents[0]['text'][:500] + "...")

# Cell 3: Text Preprocessing
"""
### Step 2: Clean and Chunk Documents

Prepare text for LLM processing by:
- Cleaning artifacts and normalizing
- Chunking into manageable pieces
- Extracting sections
"""

preprocessed_docs = []

for doc in documents:
    processed = TextCleaner.preprocess_document(doc, chunk_size=2000, overlap=200)
    preprocessed_docs.append(processed)
    
    print(f"✅ {doc['filename']}:")
    print(f"   - Cleaned text: {len(processed['cleaned_text'])} characters")
    print(f"   - Chunks: {processed['num_chunks']}")
    print(f"   - Sections: {len(processed['sections'])}")

# Cell 4: Entity Extraction with LLM
"""
### Step 3: Extract Entities and Relationships

Using an LLM to extract structured information:
- Entities: Companies, People, Products, etc.
- Relationships: OWNS, INVESTS_IN, CEO_OF, etc.

Note: Using Mock LLM for demo. Set use_real_llm=True to use actual API.
"""

# Choose LLM client
use_real_llm = False  # Set to True if you have OpenRouter API key

if use_real_llm:
    print("🤖 Using Real LLM via OpenRouter API")
    llm_client = LLMClient()
else:
    print("🤖 Using Mock LLM (no API calls)")
    llm_client = MockLLMClient()

# Initialize extractor
extractor = EntityExtractor(llm_client)

# Extract from all documents
print("\n🔍 Extracting entities and relationships...")
extractions = extractor.extract_from_documents(preprocessed_docs)

# Create global extraction (merge all documents)
global_extraction = extractor.create_global_extraction(extractions)

print(f"\n✅ Extraction Complete!")
print(f"   - Total Entities: {len(global_extraction['entities'])}")
print(f"   - Total Relationships: {len(global_extraction['relationships'])}")
print(f"   - Documents Processed: {global_extraction['num_documents']}")

# Cell 5: Display Extracted Entities
"""
### View Extracted Entities

Let's examine what entities were extracted:
"""

# Create DataFrame for better visualization
entity_df = pd.DataFrame(global_extraction['entities'])
print(f"📊 Entity Distribution by Type:")
print(entity_df['type'].value_counts())

print(f"\n📝 Sample Entities:")
display(entity_df.head(15))

# Cell 6: Display Extracted Relationships
"""
### View Extracted Relationships

Relationships connect our entities:
"""

# Create DataFrame for relationships
rel_df = pd.DataFrame(global_extraction['relationships'])
print(f"🔗 Relationship Distribution by Type:")
print(rel_df['type'].value_counts())

print(f"\n📝 Sample Relationships:")
display(rel_df.head(15))

# Cell 7: Connect to Neo4j
"""
### Step 4: Connect to Neo4j Database

Make sure Neo4j is running before executing this cell.
Update credentials if needed.
"""

# Neo4j credentials (update these!)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this!

try:
    neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_client.connect()
    print("✅ Successfully connected to Neo4j!")
    
    # Get current database statistics
    stats = neo4j_client.get_graph_statistics()
    print(f"\n📊 Current Database Stats:")
    print(f"   - Total Nodes: {stats['total_nodes']}")
    print(f"   - Total Relationships: {stats['total_relationships']}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Neo4j is running")
    print("2. Check your credentials")
    print("3. Verify the URI (bolt://localhost:7687)")

# Cell 8: Build Knowledge Graph
"""
### Step 5: Construct Knowledge Graph

Insert entities and relationships into Neo4j.
"""

# Initialize constructor
constructor = GraphConstructor(neo4j_client)

# Build graph (clear existing data)
print("🏗️  Building knowledge graph...")
print("⚠️  This will clear the existing database!")

stats = constructor.build_graph_from_extraction(
    global_extraction, 
    clear_existing=True
)

print(f"\n✅ Graph Construction Complete!")
print(f"   - Nodes Created: {stats['nodes_created']}")
print(f"   - Relationships Created: {stats['relationships_created']}")
print(f"   - Nodes Failed: {stats['nodes_failed']}")
print(f"   - Relationships Failed: {stats['relationships_failed']}")

# Get updated statistics
final_stats = neo4j_client.get_graph_statistics()
print(f"\n📊 Final Database Stats:")
print(f"   - Total Nodes: {final_stats['total_nodes']}")
print(f"   - Total Relationships: {final_stats['total_relationships']}")

print(f"\n🏷️  Node Types:")
for node_type, count in final_stats['node_types'].items():
    print(f"   - {node_type}: {count}")

print(f"\n🔗 Relationship Types:")
for rel_type, count in final_stats['relationship_types'].items():
    print(f"   - {rel_type}: {count}")

# Cell 9: Create Interactive Visualization
"""
### Step 6: Visualize Knowledge Graph

Create an interactive PyVis network visualization.
"""

visualizer = GraphVisualizer(neo4j_client)

print("🎨 Creating interactive visualization...")

# Generate PyVis network
viz_file = visualizer.create_pyvis_network("demo_graph.html")

print(f"✅ Visualization created: {viz_file}")
print("\nYou can:")
print("1. Open the HTML file in your browser")
print("2. Or display it inline below:")

# Display in notebook
display(IFrame(viz_file, width=1000, height=600))

# Cell 10: Create Plotly Visualization
"""
### Alternative: Plotly Visualization

A different style of visualization using Plotly.
"""

print("🎨 Creating Plotly visualization...")

plotly_file = visualizer.create_plotly_network("demo_plotly.html")

print(f"✅ Plotly visualization created: {plotly_file}")

# Display in notebook
display(IFrame(plotly_file, width=1000, height=600))

# Cell 11: Statistics Visualization
"""
### Graph Statistics Dashboard

Visualize entity type distributions.
"""

print("📊 Creating statistics visualization...")

stats_file = visualizer.create_statistics_visualization("demo_stats.html")

print(f"✅ Statistics created: {stats_file}")

display(IFrame(stats_file, width=1000, height=500))

# Cell 12: Subgraph Visualization
"""
### Focused Subgraph View

Visualize relationships around a specific entity.
"""

# Pick a central node (e.g., a company name)
center_node = "Samsung Electronics"  # Change this to any entity name

print(f"🔍 Creating subgraph for '{center_node}'...")

subgraph_file = visualizer.create_subgraph_visualization(
    center_node, 
    depth=2,
    output_file="demo_subgraph.html"
)

if subgraph_file:
    print(f"✅ Subgraph created: {subgraph_file}")
    display(IFrame(subgraph_file, width=1000, height=600))
else:
    print(f"❌ No data found for node: {center_node}")

# Cell 13: Natural Language Query Interface
"""
### Step 7: Query with Natural Language

Convert English questions to Cypher queries.
"""

# Initialize query interface
converter = NLToCypherConverter(llm_client, neo4j_client)
query_interface = QueryInterface(converter)

print("💬 Natural Language Query Interface Ready!")
print("\n" + query_interface.show_suggestions())

# Cell 14: Execute Sample Queries
"""
### Try Sample Queries

Execute some example natural language queries.
"""

sample_questions = [
    "Which companies did Samsung invest in?",
    "Who is the CEO of Apple?",
    "What products does Tesla develop?",
    "Show all partnerships between companies"
]

for question in sample_questions:
    print(f"\n{'='*70}")
    print(f"❓ {question}")
    print(f"{'='*70}")
    
    result = query_interface.process_query(question)
    print(result)

# Cell 15: Interactive Query Input
"""
### Interactive Query Session

Ask your own questions!
"""

print("💬 Interactive Query Mode")
print("Type your questions below (type 'exit' to stop)\n")

while True:
    question = input("Query> ").strip()
    
    if question.lower() in ['exit', 'quit', 'q', '']:
        break
    
    if question.lower() == 'history':
        print(query_interface.show_history())
        continue
    
    if question.lower() == 'suggestions':
        print(query_interface.show_suggestions())
        continue
    
    result = query_interface.process_query(question)
    print(f"\n{result}\n")

print("✅ Query session ended")

# Cell 16: Advanced Cypher Queries
"""
### Advanced Cypher Queries

Execute custom Cypher queries directly.
"""

print("🔍 Running Advanced Queries...\n")

# Query 1: Find most connected companies
query1 = """
MATCH (c:Company)-[r]-()
WITH c, count(r) as connections
ORDER BY connections DESC
LIMIT 5
RETURN c.name as company, connections
"""

print("📊 Top 5 Most Connected Companies:")
results = neo4j_client.execute_cypher(query1)
for i, record in enumerate(results, 1):
    print(f"   {i}. {record['company']}: {record['connections']} connections")

# Query 2: Find all CEO relationships
query2 = """
MATCH (p:Person)-[:CEO_OF]->(c:Company)
RETURN p.name as ceo, c.name as company
"""

print("\n👔 CEO Relationships:")
results = neo4j_client.execute_cypher(query2)
for record in results:
    print(f"   - {record['ceo']} is CEO of {record['company']}")

# Query 3: Find investment chains
query3 = """
MATCH path = (c1:Company)-[:INVESTS_IN*1..2]->(c2:Company)
RETURN c1.name as investor, c2.name as target, length(path) as hops
LIMIT 10
"""

print("\n💰 Investment Chains:")
results = neo4j_client.execute_cypher(query3)
for record in results:
    print(f"   - {record['investor']} -> {record['target']} ({record['hops']} hops)")

# Cell 17: Export Graph Data
"""
### Export Data

Save graph data for further analysis.
"""

# Export as JSON
json_file = visualizer.export_graph_data("demo_export.json")
print(f"✅ Graph data exported to: {json_file}")

# Load and display summary
with open(json_file, 'r') as f:
    graph_data = json.load(f)

print(f"\n📦 Export Summary:")
print(f"   - Nodes: {len(graph_data['nodes'])}")
print(f"   - Edges: {len(graph_data['edges'])}")

# Cell 18: Graph Analytics
"""
### Graph Analytics

Compute graph metrics and statistics.
"""

import networkx as nx

# Build NetworkX graph from exported data
G = nx.Graph()

for node in graph_data['nodes']:
    G.add_node(node['id'], **node['properties'])

for edge in graph_data['edges']:
    G.add_edge(edge['source'], edge['target'], type=edge['type'])

print("📈 Graph Analytics:\n")

# Basic metrics
print(f"Number of Nodes: {G.number_of_nodes()}")
print(f"Number of Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")

# Connected components
num_components = nx.number_connected_components(G)
print(f"Connected Components: {num_components}")

# Centrality measures
print("\n🌟 Top 5 Most Central Nodes (Degree Centrality):")
degree_centrality = nx.degree_centrality(G)
top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

for i, (node, score) in enumerate(top_central, 1):
    print(f"   {i}. {node}: {score:.4f}")

# Betweenness centrality
print("\n🌉 Top 5 Bridge Nodes (Betweenness Centrality):")
betweenness = nx.betweenness_centrality(G)
top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

for i, (node, score) in enumerate(top_between, 1):
    print(f"   {i}. {node}: {score:.4f}")

# Cell 19: Cleanup
"""
### Cleanup and Summary

Final statistics and cleanup.
"""

print("📊 Final Summary:\n")

# Get final statistics
final_stats = neo4j_client.get_graph_statistics()

print(f"✅ Knowledge Graph Statistics:")
print(f"   - Documents Processed: {len(documents)}")
print(f"   - Entities Extracted: {len(global_extraction['entities'])}")
print(f"   - Relationships Extracted: {len(global_extraction['relationships'])}")
print(f"   - Nodes in Database: {final_stats['total_nodes']}")
print(f"   - Relationships in Database: {final_stats['total_relationships']}")

print(f"\n📁 Generated Files:")
print(f"   - Interactive Graph: {viz_file}")
print(f"   - Plotly Graph: {plotly_file}")
print(f"   - Statistics: {stats_file}")
print(f"   - JSON Export: {json_file}")

print("\n🎉 Demo Complete! You've successfully:")
print("   ✅ Extracted entities from financial documents")
print("   ✅ Built a knowledge graph in Neo4j")
print("   ✅ Created interactive visualizations")
print("   ✅ Queried the graph with natural language")
print("   ✅ Performed graph analytics")

# Close connection
neo4j_client.close()
print("\n👋 Neo4j connection closed")

# Cell 20: Next Steps
"""
### 🚀 Next Steps

Ideas for extending this project:

1. **Add More Documents**: Process real financial reports
2. **Improve Extraction**: Fine-tune prompts or use better models
3. **Temporal Analysis**: Track changes over time
4. **Sentiment Analysis**: Add sentiment to financial metrics
5. **Recommendation System**: Suggest similar companies or investments
6. **Web Interface**: Deploy with Streamlit or Flask
7. **Real-time Updates**: Monitor news feeds for graph updates
8. **Advanced Analytics**: PageRank, community detection
9. **Multi-language Support**: Process documents in various languages
10. **API Development**: Create REST API for graph queries

### 📚 Resources

- Neo4j Documentation: https://neo4j.com/docs/
- OpenRouter API: https://openrouter.ai/docs
- NetworkX: https://networkx.org/
- PyVis: https://pyvis.readthedocs.io/

### 🤝 Contributing

Fork this project and add your own features!
"""

print("📖 See cell output for next steps and resources")