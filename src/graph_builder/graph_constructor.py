"""
Graph Constructor Module
Builds knowledge graph from extracted entities and relationships
WITH: Enhanced duplicate handling and metadata support
"""
from typing import Dict, List
import logging
from src.graph_builder.neo4j_client import Neo4jClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphConstructor:
    """Construct knowledge graph from extracted data"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph constructor
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self.client = neo4j_client
    
    def build_graph_from_extraction(self, extraction: Dict, 
                                   clear_existing: bool = False) -> Dict:
        """
        Build graph from a single extraction result
        NOW HANDLES: original_names and metadata from enhanced extraction
        
        Args:
            extraction: Dictionary with entities and relationships
            clear_existing: Whether to clear existing graph
            
        Returns:
            Statistics about graph construction
        """
        if clear_existing:
            logger.warning("Clearing existing graph database")
            self.client.clear_database()
            self.client.create_constraints()
            self.client.create_indexes()
        
        stats = {
            'nodes_created': 0,
            'nodes_failed': 0,
            'relationships_created': 0,
            'relationships_failed': 0,
            'nodes_merged': 0
        }
        
        # Create nodes
        entities = extraction.get('entities', [])
        logger.info(f"Creating {len(entities)} nodes")
        
        for entity in tqdm(entities, desc="Creating nodes"):
            name = entity.get('name', '')
            entity_type = entity.get('type', 'Unknown')
            properties = entity.get('properties', {}).copy()
            
            # NEW: Include original_names if present
            if 'original_names' in entity:
                properties['original_names'] = entity['original_names']
            
            # NEW: Include source metadata
            if 'source_document' in entity:
                properties['source_document'] = entity['source_document']
            if 'source_type' in entity:
                properties['source_type'] = entity['source_type']
            
            if name:
                success = self.client.create_node(name, entity_type, properties)
                if success:
                    stats['nodes_created'] += 1
                else:
                    stats['nodes_failed'] += 1
        
        # Create relationships
        relationships = extraction.get('relationships', [])
        logger.info(f"Creating {len(relationships)} relationships")
        
        for rel in tqdm(relationships, desc="Creating relationships"):
            source = rel.get('source', '')
            target = rel.get('target', '')
            rel_type = rel.get('type', 'RELATED_TO')
            properties = rel.get('properties', {})
            
            if source and target:
                success = self.client.create_relationship(
                    source, target, rel_type, properties
                )
                if success:
                    stats['relationships_created'] += 1
                else:
                    stats['relationships_failed'] += 1
        
        logger.info(f"Graph construction complete: {stats}")
        return stats
    
    def build_graph_from_extractions(self, extractions: List[Dict],
                                    clear_existing: bool = False) -> Dict:
        """
        Build graph from multiple extraction results
        
        Args:
            extractions: List of extraction dictionaries
            clear_existing: Whether to clear existing graph
            
        Returns:
            Combined statistics
        """
        if clear_existing:
            logger.warning("Clearing existing graph database")
            self.client.clear_database()
            self.client.create_constraints()
            self.client.create_indexes()
        
        combined_stats = {
            'nodes_created': 0,
            'nodes_failed': 0,
            'relationships_created': 0,
            'relationships_failed': 0,
            'documents_processed': len(extractions)
        }
        
        for i, extraction in enumerate(extractions):
            logger.info(f"Processing extraction {i + 1}/{len(extractions)}")
            
            # Don't clear after first iteration
            stats = self.build_graph_from_extraction(extraction, clear_existing=False)
            
            combined_stats['nodes_created'] += stats['nodes_created']
            combined_stats['nodes_failed'] += stats['nodes_failed']
            combined_stats['relationships_created'] += stats['relationships_created']
            combined_stats['relationships_failed'] += stats['relationships_failed']
        
        return combined_stats
    
    def enrich_graph(self):
        """
        Add computed relationships and properties to the graph
        For example, infer competitor relationships, calculate centrality, etc.
        """
        logger.info("Enriching graph with computed relationships")
        
        # Example: Find companies that share many connections (potential competitors)
        competitor_query = """
        MATCH (c1:Company)-[]-(n)-[]-(c2:Company)
        WHERE c1 <> c2 AND NOT (c1)-[:COMPETES_WITH]-(c2)
        WITH c1, c2, count(DISTINCT n) as shared_connections
        WHERE shared_connections >= 2
        MERGE (c1)-[r:POTENTIAL_COMPETITOR {shared_connections: shared_connections}]->(c2)
        RETURN count(r) as new_relationships
        """
        
        try:
            result = self.client.execute_cypher(competitor_query)
            if result:
                logger.info(f"Added {result[0]['new_relationships']} competitor relationships")
        except Exception as e:
            logger.error(f"Error enriching graph: {e}")
    
    def create_summary_nodes(self):
        """
        Create high-level summary nodes for analytics
        """
        logger.info("Creating summary nodes")
        
        # Create industry summary nodes
        industry_query = """
        MATCH (c:Company)
        WHERE c.industry IS NOT NULL
        WITH c.industry as industry, collect(c) as companies
        CREATE (s:IndustrySummary {
            name: industry,
            company_count: size(companies)
        })
        WITH s, companies
        UNWIND companies as company
        MERGE (company)-[:BELONGS_TO_INDUSTRY]->(s)
        RETURN count(s) as summaries_created
        """
        
        try:
            result = self.client.execute_cypher(industry_query)
            if result:
                logger.info(f"Created {result[0]['summaries_created']} industry summaries")
        except Exception as e:
            logger.warning(f"Could not create summary nodes: {e}")
    
    def validate_graph(self) -> Dict:
        """
        Validate graph structure and return issues
        NOW INCLUDES: Enhanced duplicate detection using original_names
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating graph structure")
        
        issues = {
            'isolated_nodes': [],
            'missing_relationships': [],
            'duplicate_nodes': [],
            'abstract_nodes': []
        }
        
        # Find isolated nodes (no relationships)
        isolated_query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN n.name as name, labels(n)[0] as label
        LIMIT 20
        """
        
        result = self.client.execute_cypher(isolated_query)
        issues['isolated_nodes'] = result
        
        # Check for potential duplicate nodes (similar names)
        duplicate_query = """
        MATCH (n1), (n2)
        WHERE id(n1) < id(n2)
        AND labels(n1) = labels(n2)
        AND toLower(n1.name) = toLower(n2.name)
        AND n1.name <> n2.name
        RETURN n1.name as name1, n2.name as name2, labels(n1)[0] as label
        LIMIT 20
        """
        
        result = self.client.execute_cypher(duplicate_query)
        issues['duplicate_nodes'] = result
        
        # NEW: Find abstract concept nodes (should have been filtered)
        abstract_query = """
        MATCH (n)
        WHERE toLower(n.name) IN ['revenue', 'profit', 'growth', 'income', 'expense', 'the company', 'the product']
        RETURN n.name as name, labels(n)[0] as label
        LIMIT 20
        """
        
        result = self.client.execute_cypher(abstract_query)
        issues['abstract_nodes'] = result
        
        logger.info(f"Validation complete: Found {len(issues['isolated_nodes'])} isolated nodes, "
                   f"{len(issues['duplicate_nodes'])} potential duplicates, "
                   f"{len(issues['abstract_nodes'])} abstract nodes")
        
        return issues
    
    # NEW: Clean up abstract nodes
    def cleanup_abstract_nodes(self) -> int:
        """
        Remove abstract concept nodes that shouldn't be in the graph
        
        Returns:
            Number of nodes removed
        """
        logger.info("Cleaning up abstract concept nodes")
        
        cleanup_query = """
        MATCH (n)
        WHERE toLower(n.name) IN ['revenue', 'profit', 'growth', 'income', 'expense', 
                                   'cost', 'sales', 'the company', 'the product', 'the person',
                                   'increase', 'decrease', 'report', 'document', 'statement']
        DETACH DELETE n
        RETURN count(n) as removed
        """
        
        try:
            result = self.client.execute_cypher(cleanup_query)
            if result:
                removed = result[0].get('removed', 0)
                logger.info(f"Removed {removed} abstract concept nodes")
                return removed
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up abstract nodes: {e}")
            return 0
    
    # NEW: Merge similar nodes based on original_names
    def merge_similar_nodes(self) -> int:
        """
        Merge nodes that have overlapping original_names
        
        Returns:
            Number of nodes merged
        """
        logger.info("Merging similar nodes based on original_names")
        
        merge_query = """
        MATCH (n1), (n2)
        WHERE id(n1) < id(n2)
        AND labels(n1) = labels(n2)
        AND any(name1 IN n1.original_names WHERE name1 IN n2.original_names)
        
        // Keep the node with more relationships
        WITH n1, n2, 
             size((n1)--()) as n1_rels, 
             size((n2)--()) as n2_rels
        WITH CASE WHEN n1_rels >= n2_rels THEN n1 ELSE n2 END as keep,
             CASE WHEN n1_rels >= n2_rels THEN n2 ELSE n1 END as remove
        
        // Transfer relationships
        OPTIONAL MATCH (remove)-[r1]->(target)
        WHERE target <> keep
        FOREACH (_ IN CASE WHEN target IS NOT NULL THEN [1] ELSE [] END |
            MERGE (keep)-[r2:type(r1)]->(target)
            SET r2 += properties(r1)
        )
        
        OPTIONAL MATCH (source)-[r3]->(remove)
        WHERE source <> keep
        FOREACH (_ IN CASE WHEN source IS NOT NULL THEN [1] ELSE [] END |
            MERGE (source)-[r4:type(r3)]->(keep)
            SET r4 += properties(r3)
        )
        
        // Merge properties
        SET keep.original_names = COALESCE(keep.original_names, []) + 
                                   COALESCE(remove.original_names, [])
        
        // Delete duplicate
        DETACH DELETE remove
        
        RETURN count(remove) as merged
        """
        
        try:
            result = self.client.execute_cypher(merge_query)
            if result:
                merged = result[0].get('merged', 0)
                logger.info(f"Merged {merged} duplicate nodes")
                return merged
            return 0
        except Exception as e:
            logger.error(f"Error merging similar nodes: {e}")
            return 0