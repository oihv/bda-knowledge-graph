#!/usr/bin/env python3
"""
Test Script for Cypher Query Patterns
Validates that our generated Cypher patterns work with the actual database
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.graph_builder.neo4j_client import Neo4jClient
import config

def test_query_patterns():
    """Test common Cypher query patterns"""
    
    client = Neo4jClient()
    client.connect()
    
    test_queries = [
        {
            'name': 'Count Companies',
            'query': 'MATCH (c:Company) RETURN count(c) as count',
            'expected': 'count > 0'
        },
        {
            'name': 'Apple Products',
            'query': 'MATCH (c:Company)-[:DEVELOPS|OWNS]->(p:Product) WHERE toLower(c.name) CONTAINS toLower("apple") RETURN p.name as product LIMIT 5',
            'expected': 'at least some products'
        },
        {
            'name': 'Apple Financial Metrics',
            'query': 'MATCH (c:Company)-[:HAS_METRIC]->(fm:FinancialMetric) WHERE toLower(c.name) CONTAINS toLower("apple") RETURN fm.name as metric LIMIT 5',
            'expected': 'financial metrics'
        },
        {
            'name': 'Apple Locations',
            'query': 'MATCH (c:Company)-[:LOCATED_IN]->(l:Location) WHERE toLower(c.name) CONTAINS toLower("apple") RETURN l.name as location',
            'expected': 'location data'
        },
        {
            'name': 'Apple Technologies',
            'query': 'MATCH (c:Company)-[:DEVELOPS]->(t:Technology) WHERE toLower(c.name) CONTAINS toLower("apple") RETURN t.name as technology LIMIT 5',
            'expected': 'technologies'
        },
        {
            'name': 'Apple Competitors',
            'query': 'MATCH (c:Company)-[:COMPETES_WITH]->(comp) WHERE toLower(c.name) CONTAINS toLower("apple") RETURN comp.name as competitor LIMIT 5',
            'expected': 'competitors'
        },
        {
            'name': 'Most Connected Entities',
            'query': 'MATCH (n) WITH n, [(n)--() | 1] as connections WHERE size(connections) > 5 RETURN labels(n)[0] as type, n.name as entity, size(connections) as connections ORDER BY connections DESC LIMIT 5',
            'expected': 'connected entities'
        }
    ]
    
    print("Testing Cypher Query Patterns")
    print("=" * 50)
    
    all_passed = True
    
    for test in test_queries:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"Query: {test['query']}")
        
        try:
            results = client.execute_cypher(test['query'])
            print(f"✅ Success: {len(results)} results")
            
            # Show sample results
            if results:
                for i, result in enumerate(results[:3]):
                    print(f"   {i+1}. {result}")
            else:
                print("   (No results found)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            all_passed = False
    
    client.close()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All query patterns executed successfully!")
    else:
        print("❌ Some query patterns failed.")
    
    return all_passed

if __name__ == "__main__":
    test_query_patterns()