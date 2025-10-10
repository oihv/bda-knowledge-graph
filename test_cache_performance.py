#!/usr/bin/env python3
"""
Test script to verify caching performance improvements
"""
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_interface.nl_to_cypher import NLToCypherConverter
from src.graph_builder.neo4j_client import Neo4jClient
from src.llm_extraction.llm_client import MockLLMClient


def test_caching_performance():
    """Test the caching performance improvements"""
    print("Testing NL-to-Cypher caching performance...")
    
    # Initialize clients with hardcoded values to avoid config dependency
    neo4j_client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    mock_llm = MockLLMClient()
    converter = NLToCypherConverter(mock_llm, neo4j_client, cache_ttl=10)  # 10 second cache
    
    try:
        # Test database connection
        stats = neo4j_client.get_graph_statistics()
        print(f"✅ Connected to Neo4j: {stats.get('total_nodes', 0)} nodes, {stats.get('total_relationships', 0)} relationships")
        
        # First call - should hit database
        print("\n1. First call (should hit database):")
        start_time = time.time()
        schema1 = converter.get_schema_context()
        time1 = time.time() - start_time
        print(f"   Time: {time1:.3f} seconds")
        print(f"   Schema length: {len(schema1)} characters")
        
        # Second call - should use cache
        print("\n2. Second call (should use cache):")
        start_time = time.time()
        schema2 = converter.get_schema_context()
        time2 = time.time() - start_time
        print(f"   Time: {time2:.3f} seconds")
        print(f"   Schema length: {len(schema2)} characters")
        print(f"   Same content: {schema1 == schema2}")
        
        # Performance improvement
        if time1 > 0:
            improvement = ((time1 - time2) / time1) * 100
            print(f"   Speed improvement: {improvement:.1f}%")
        
        # Test cache invalidation  
        print("\n3. After cache invalidation (should hit database again):")
        converter.invalidate_cache()
        start_time = time.time()
        schema3 = converter.get_schema_context()
        time3 = time.time() - start_time
        print(f"   Time: {time3:.3f} seconds")
        print(f"   Same content: {schema1 == schema3}")
        
        # Test suggestions performance
        print("\n4. Testing suggestions performance:")
        start_time = time.time()
        suggestions = converter.get_suggested_questions()
        time4 = time.time() - start_time
        print(f"   Time: {time4:.3f} seconds")
        print(f"   Number of suggestions: {len(suggestions)}")
        
        print("\n✅ Cache performance test completed successfully!")
        print("\nPerformance Summary:")
        print(f"   Database calls: {time1:.3f}s → {time2:.3f}s (cached)")
        print(f"   Speed improvement: {((time1 - time2) / time1) * 100:.1f}%" if time1 > 0 else "   N/A")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    finally:
        neo4j_client.close()
    
    return True


if __name__ == "__main__":
    success = test_caching_performance()
    sys.exit(0 if success else 1)