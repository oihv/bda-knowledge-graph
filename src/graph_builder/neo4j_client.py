"""
Neo4j Client Module
Handles connection and operations with Neo4j graph database
"""
from neo4j import GraphDatabase
from typing import Dict, List, Optional
import logging
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j graph database operations"""
    
    def __init__(self, uri: str = NEO4J_URI, 
                 user: str = NEO4J_USER, 
                 password: str = NEO4J_PASSWORD):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_constraints(self):
        """Create uniqueness constraints for entity types"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
        
        logger.info("Database constraints created")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        
        logger.info("Database indexes created")
    
    def create_node(self, name: str, label: str, properties: Dict = None) -> bool:
        """
        Create or update a node in the graph
        
        Args:
            name: Node name (unique identifier)
            label: Node label (type)
            properties: Additional properties
            
        Returns:
            True if successful
        """
        props = properties or {}
        props['name'] = name
        
        # Convert properties to Cypher-compatible format
        props_str = ', '.join([f"{k}: ${k}" for k in props.keys()])
        
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += {{{props_str}}}
        RETURN n
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, **props)
            return True
        except Exception as e:
            logger.error(f"Error creating node {name}: {e}")
            return False
    
    def create_relationship(self, source: str, target: str, 
                          rel_type: str, properties: Dict = None) -> bool:
        """
        Create a relationship between two nodes
        
        Args:
            source: Source node name
            target: Target node name
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            True if successful
        """
        props = properties or {}
        
        # Build properties string
        if props:
            props_str = ', '.join([f"{k}: ${k}" for k in props.keys()])
            props_clause = f"{{{props_str}}}"
        else:
            props_clause = ""
        
        query = f"""
        MATCH (source {{name: $source}})
        MATCH (target {{name: $target}})
        MERGE (source)-[r:{rel_type} {props_clause}]->(target)
        RETURN r
        """
        
        try:
            with self.driver.session() as session:
                params = {'source': source, 'target': target}
                params.update(props)
                result = session.run(query, **params)
                if result.single():
                    return True
                return False
        except Exception as e:
            logger.error(f"Error creating relationship {source}-[{rel_type}]->{target}: {e}")
            return False
    
    def get_node(self, name: str) -> Optional[Dict]:
        """
        Retrieve a node by name
        
        Args:
            name: Node name
            
        Returns:
            Node properties or None
        """
        query = "MATCH (n {name: $name}) RETURN n"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, name=name)
                record = result.single()
                if record:
                    return dict(record['n'])
                return None
        except Exception as e:
            logger.error(f"Error retrieving node {name}: {e}")
            return None
    
    def get_relationships(self, node_name: str, direction: str = "both") -> List[Dict]:
        """
        Get all relationships for a node
        
        Args:
            node_name: Node name
            direction: 'in', 'out', or 'both'
            
        Returns:
            List of relationship dictionaries
        """
        if direction == "out":
            query = """
            MATCH (n {name: $name})-[r]->(m)
            RETURN type(r) as type, m.name as target, properties(r) as props
            """
        elif direction == "in":
            query = """
            MATCH (n {name: $name})<-[r]-(m)
            RETURN type(r) as type, m.name as source, properties(r) as props
            """
        else:  # both
            query = """
            MATCH (n {name: $name})-[r]-(m)
            RETURN type(r) as type, m.name as connected, properties(r) as props
            """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, name=node_name)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving relationships for {node_name}: {e}")
            return []
    
    def execute_cypher(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        Execute a custom Cypher query safely and normalize results.

        Args:
            query: Cypher query string
            parameters: Query parameters (optional)

        Returns:
            List of dictionaries with JSON-serializable node, relationship, or value data.
        """
        params = parameters or {}

        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                data = []

                for record in result:
                    parsed_record = {}
                    for key, value in record.items():
                        # --- Neo4j Node ---
                        if hasattr(value, "items"):  # dict-like Node
                            parsed_record[key] = dict(value.items())

                        elif hasattr(value, "labels"):  # Node object
                            parsed_record[key] = {
                                **dict(value),
                                "_labels": list(value.labels)
                            }

                        # --- Neo4j Relationship ---
                        elif hasattr(value, "type") and hasattr(value, "nodes"):
                            parsed_record[key] = {
                                "type": value.type,
                                "start_node": dict(value.start_node.items()),
                                "end_node": dict(value.end_node.items()),
                                "properties": dict(value.items())
                            }

                        # --- Lists of Nodes or Relationships ---
                        elif isinstance(value, list):
                            parsed_list = []
                            for v in value:
                                if hasattr(v, "items"):
                                    parsed_list.append(dict(v.items()))
                                elif hasattr(v, "labels"):
                                    parsed_list.append({
                                        **dict(v),
                                        "_labels": list(v.labels)
                                    })
                                elif hasattr(v, "type") and hasattr(v, "nodes"):
                                    parsed_list.append({
                                        "type": v.type,
                                        "start_node": dict(v.start_node.items()),
                                        "end_node": dict(v.end_node.items()),
                                        "properties": dict(v.items())
                                    })
                                else:
                                    parsed_list.append(v)
                            parsed_record[key] = parsed_list

                        # --- Fallback for primitives or strings ---
                        else:
                            parsed_record[key] = str(value)

                    data.append(parsed_record)
                return data

        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []

    
    def get_graph_statistics(self) -> Dict:
        """
        Get comprehensive and safe graph statistics.

        Returns:
            Dictionary containing total counts, node type counts,
            and relationship type counts.
        """
        queries = {
            'total_nodes': "MATCH (n) RETURN count(n) AS count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) AS count",
            'node_types': "MATCH (n) RETURN coalesce(labels(n)[0], 'Unknown') AS label, count(*) AS count",
            'relationship_types': "MATCH ()-[r]->() RETURN coalesce(type(r), 'Unknown') AS type, count(*) AS count"
        }

        stats = {
            'total_nodes': 0,
            'total_relationships': 0,
            'node_types': {},
            'relationship_types': {}
        }

        try:
            with self.driver.session() as session:
                # --- Total nodes ---
                try:
                    result = session.run(queries['total_nodes']).single()
                    stats['total_nodes'] = int(result['count']) if result and 'count' in result else 0
                except Exception as e:
                    logger.warning(f"Failed to count nodes: {e}")

                # --- Total relationships ---
                try:
                    result = session.run(queries['total_relationships']).single()
                    stats['total_relationships'] = int(result['count']) if result and 'count' in result else 0
                except Exception as e:
                    logger.warning(f"Failed to count relationships: {e}")

                # --- Node types ---
                try:
                    result = session.run(queries['node_types'])
                    for record in result:
                        label = record.get('label', 'Unknown')
                        count = int(record.get('count', 0))
                        stats['node_types'][label] = count
                except Exception as e:
                    logger.warning(f"Failed to get node types: {e}")

                # --- Relationship types ---
                try:
                    result = session.run(queries['relationship_types'])
                    for record in result:
                        rel_type = record.get('type', 'Unknown')
                        count = int(record.get('count', 0))
                        stats['relationship_types'][rel_type] = count
                except Exception as e:
                    logger.warning(f"Failed to get relationship types: {e}")

            logger.info(f"Graph statistics retrieved: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error fetching graph statistics: {e}")
            return stats
