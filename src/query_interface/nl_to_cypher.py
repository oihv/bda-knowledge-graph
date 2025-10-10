"""
Natural Language to Cypher Query Module
Convert natural language questions to Cypher queries
"""
from typing import Dict, List, Optional
import logging
import time
from src.llm_extraction.llm_client import LLMClient
from src.graph_builder.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLToCypherConverter:
    """Convert natural language queries to Cypher"""
    
    def __init__(self, llm_client: LLMClient, neo4j_client: Neo4jClient, cache_ttl: int = 300):
        """
        Initialize converter
        
        Args:
            llm_client: LLM client for query generation
            neo4j_client: Neo4j client for execution
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.llm_client = llm_client
        self.neo4j_client = neo4j_client
        self.neo4j_client.connect()  # Ensure connection
        self._schema_cache = None
        self._cache_timestamp = None
        self._cache_ttl = cache_ttl
        
        # Pre-built query templates to avoid API calls
        self.query_templates = {
            'count companies': "MATCH (c:Company) RETURN count(c) as company_count",
            'count all companies': "MATCH (c:Company) RETURN count(c) as company_count", 
            'list companies': "MATCH (c:Company) RETURN c.name as company LIMIT 20",
            'show companies': "MATCH (c:Company) RETURN c.name as company LIMIT 20",
            'count products': "MATCH (p:Product) RETURN count(p) as product_count",
            'list products': "MATCH (p:Product) RETURN p.name as product LIMIT 20",
            'count people': "MATCH (p:Person) RETURN count(p) as person_count",
            'list people': "MATCH (p:Person) RETURN p.name as person LIMIT 20",
            'count locations': "MATCH (l:Location) RETURN count(l) as location_count",
            'list locations': "MATCH (l:Location) RETURN l.name as location LIMIT 20",
            'count relationships': "MATCH ()-[r]->() RETURN count(r) as relationship_count",
            'show relationship types': "MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_type",
            'show node types': "MATCH (n) RETURN DISTINCT labels(n)[0] as node_type",
            'most connected': "MATCH (n) WITH n, [(n)--() | 1] as connections WHERE size(connections) > 0 RETURN labels(n)[0] as type, n.name as entity, size(connections) as connections ORDER BY connections DESC LIMIT 10",
            'apple products': "MATCH (c:Company)-[:DEVELOPS|:OWNS]->(p:Product) WHERE toLower(c.name) CONTAINS 'apple' RETURN p.name as product LIMIT 20",
            'apple locations': "MATCH (c:Company)-[:LOCATED_IN]->(l:Location) WHERE toLower(c.name) CONTAINS 'apple' RETURN l.name as location",
            'apple relationships': "MATCH (c:Company)-[r]->(n) WHERE toLower(c.name) CONTAINS 'apple' RETURN type(r) as relationship, labels(n)[0] as target_type, n.name as target LIMIT 20"
        }
    
    def get_schema_context(self) -> str:
        """
        Get graph schema information for context (with caching)
        
        Returns:
            Schema description string
        """
        # Check if cache is valid
        current_time = time.time()
        if (self._schema_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            logger.debug("Using cached schema context")
            return self._schema_cache
        
        logger.info("Refreshing schema cache")
        
        # Generate fresh schema context
        stats = self.neo4j_client.get_graph_statistics()
        
        node_types = list(stats.get('node_types', {}).keys())
        rel_types = list(stats.get('relationship_types', {}).keys())
        
        # Get sample entities for each type
        sample_entities = {}
        for node_type in node_types[:5]:  # Limit to top 5 types
            try:
                query = f"MATCH (n:{node_type}) RETURN n.name as name LIMIT 3"
                results = self.neo4j_client.execute_cypher(query)
                if results:
                    sample_entities[node_type] = [r['name'] for r in results]
            except:
                sample_entities[node_type] = []
        
        schema = f"""
Graph Schema Information:
- Node Types ({len(node_types)}): {', '.join(node_types)}
- Relationship Types ({len(rel_types)}): {', '.join(rel_types)}

Sample Entities:"""
        
        for node_type, samples in sample_entities.items():
            if samples:
                schema += f"\n- {node_type}: {', '.join(samples[:3])}"
        
        schema += """

Important Properties:
- All nodes have a 'name' property (primary identifier)
- Companies may have additional properties like address, phone, etc.
- Products may have properties like trading_symbol, par_value, etc.
- FinancialMetrics contain business measurements and KPIs

Query Guidelines:
- Use case-insensitive matching: WHERE toLower(n.name) CONTAINS toLower('search_term')
- Common patterns: (Company)-[:OWNS|DEVELOPS]->(Product)
- Geographic: (Entity)-[:LOCATED_IN]->(Location)
- Business: (Company)-[:COMPETES_WITH|PARTNERS_WITH]->(Company)
"""
        
        # Cache the result
        self._schema_cache = schema
        self._cache_timestamp = current_time
        
        return schema
    
    def invalidate_cache(self):
        """Manually invalidate the schema cache"""
        logger.info("Cache invalidated manually")
        self._schema_cache = None
        self._cache_timestamp = None
    
    def refresh_cache(self) -> str:
        """Force refresh of schema cache and return new schema"""
        self.invalidate_cache()
        return self.get_schema_context()
    
    def create_cypher_prompt(self, nl_question: str) -> str:
        """Create prompt for LLM to generate Cypher query"""
        schema = self.get_schema_context()

        prompt = f"""You are an expert Neo4j Cypher query generator. Generate ONLY a valid Cypher query based on the schema and question below.

{schema}

QUERY GENERATION RULES:
1. Use proper Cypher syntax: MATCH, WHERE, RETURN
2. Use node labels and aliases: (c:Company), (p:Product)
3. Case-insensitive matching: WHERE toLower(n.name) CONTAINS toLower('search_term')
4. Limit results: LIMIT 20 (unless asking for counts)
5. Return meaningful fields, not whole nodes
6. Use appropriate relationship types from the schema

FINANCIAL DOMAIN EXAMPLES:
Q: "What products does Apple make?"
A: MATCH (c:Company)-[:DEVELOPS|:OWNS]->(p:Product) WHERE toLower(c.name) CONTAINS toLower('apple') RETURN p.name as product LIMIT 20

Q: "Show Apple's financial metrics"
A: MATCH (c:Company)-[:HAS_METRIC]->(fm:FinancialMetric) WHERE toLower(c.name) CONTAINS toLower('apple') RETURN fm.name as metric LIMIT 20

Q: "Where is Apple located?"
A: MATCH (c:Company)-[:LOCATED_IN]->(l:Location) WHERE toLower(c.name) CONTAINS toLower('apple') RETURN l.name as location

Q: "What technologies does Apple develop?"
A: MATCH (c:Company)-[:DEVELOPS]->(t:Technology) WHERE toLower(c.name) CONTAINS toLower('apple') RETURN t.name as technology LIMIT 20

Q: "Who does Apple compete with?"
A: MATCH (c:Company)-[:COMPETES_WITH]->(comp) WHERE toLower(c.name) CONTAINS toLower('apple') RETURN comp.name as competitor LIMIT 20

Q: "Count all companies"
A: MATCH (c:Company) RETURN count(c) as company_count

Q: "Most connected entities"
A: MATCH (n) WITH n, [(n)--() | 1] as connections WHERE size(connections) > 0 RETURN labels(n)[0] as type, n.name as entity, size(connections) as connections ORDER BY connections DESC LIMIT 10

Now generate the Cypher query for: "{nl_question}"

Return ONLY the Cypher query, no explanations."""
        return prompt

    
    def find_template_match(self, question: str) -> Optional[str]:
        """Check if question matches a pre-built template"""
        question_lower = question.lower().strip()
        
        # Direct exact matches
        if question_lower in self.query_templates:
            logger.info(f"Found exact template match for: {question}")
            return self.query_templates[question_lower]
        
        # Fuzzy matching for common patterns
        for template_key, cypher in self.query_templates.items():
            # Check if all key words from template are in question
            template_words = template_key.split()
            if all(word in question_lower for word in template_words):
                logger.info(f"Found fuzzy template match '{template_key}' for: {question}")
                return cypher
        
        return None

    def generate_cypher(self, question: str) -> Optional[str]:
        """Generate Cypher query from natural language using LLM API"""
        logger.info(f"Generating Cypher for: {question}")
        
        # Force all queries to use LLM API - no template matching
        logger.info("Using LLM API for Cypher generation (template matching disabled)")
        prompt = self.create_cypher_prompt(question)
        
        # Increased retries since we're forcing API usage
        max_retries = 3
        delay = 2  # Start with 2 seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting LLM Cypher generation (attempt {attempt + 1}/{max_retries})")
                response = self.llm_client.call_llm(prompt, temperature=0.1)
                if response:
                    logger.info("LLM Cypher generation successful")
                    cypher = response.strip()
                    if cypher.startswith("```"):
                        cypher = "\n".join(cypher.split("\n")[1:-1])
                    return cypher.strip()
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt+1}: {e}")
            
            if attempt < max_retries - 1:  # Don't sleep after last attempt
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay = min(delay * 2, 30)  # Exponential backoff, cap at 30s
        
        logger.error("Failed to generate Cypher after retries")
        return None

    
    def execute_natural_query(self, question: str) -> Dict:
        """
        Execute natural language query and return results
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with query, results, and metadata
        """
        # Generate Cypher
        cypher = self.generate_cypher(question)
        
        if not cypher:
            return {
                'success': False,
                'question': question,
                'error': 'Failed to generate Cypher query'
            }
        
        # Execute query
        try:
            results = self.neo4j_client.execute_cypher(cypher)
            
            return {
                'success': True,
                'question': question,
                'cypher': cypher,
                'results': results,
                'num_results': len(results)
            }
        
        except Exception as e:
            logger.error(f"Error executing Cypher: {e}")
            return {
                'success': False,
                'question': question,
                'cypher': cypher,
                'error': str(e)
            }
    
    def format_results(self, query_result: Dict) -> str:
        """
        Format query results as readable text
        
        Args:
            query_result: Result from execute_natural_query
            
        Returns:
            Formatted string
        """
        if not query_result.get('success'):
            return f"Error: {query_result.get('error', 'Unknown error')}"
        
        output = f"Question: {query_result['question']}\n"
        output += f"Generated Query: {query_result['cypher']}\n\n"
        output += f"Results ({query_result['num_results']} found):\n"
        output += "-" * 50 + "\n"
        
        results = query_result.get('results', [])
        
        if not results:
            output += "No results found.\n"
        else:
            for i, record in enumerate(results, 1):
                output += f"{i}. "
                output += ", ".join([f"{k}: {v}" for k, v in record.items()])
                output += "\n"
        
        return output
    
    def get_suggested_questions(self) -> List[str]:
        """
        Generate suggested questions for LLM-powered query generation
        
        Returns:
            List of example questions (all use LLM API)
        """
        # Sample questions that work well with LLM generation
        suggestions = [
            "Count all companies",
            "List companies", 
            "Count products",
            "List products",
            "Count people",
            "List people",
            "Count locations", 
            "Show relationship types",
            "Show node types",
            "Most connected entities",
            "Apple products",
            "Apple locations",
            "Apple relationships"
        ]
        
        return suggestions


class QueryInterface:
    """Interactive query interface"""
    
    def __init__(self, converter: NLToCypherConverter):
        """
        Initialize interface
        
        Args:
            converter: NL to Cypher converter
        """
        self.converter = converter
        self.query_history = []
    
    def process_query(self, question: str) -> str:
        """
        Process a natural language query
        
        Args:
            question: User question
            
        Returns:
            Formatted results
        """
        result = self.converter.execute_natural_query(question)
        self.query_history.append(result)
        
        return self.converter.format_results(result)
    
    def show_suggestions(self) -> str:
        """Show suggested questions"""
        suggestions = self.converter.get_suggested_questions()
        
        output = "Suggested Questions:\n"
        output += "=" * 50 + "\n"
        for i, question in enumerate(suggestions, 1):
            output += f"{i}. {question}\n"
        
        return output
    
    def show_history(self) -> str:
        """Show query history"""
        if not self.query_history:
            return "No query history yet."
        
        output = "Query History:\n"
        output += "=" * 50 + "\n"
        
        for i, result in enumerate(self.query_history, 1):
            output += f"{i}. {result['question']}\n"
            output += f"   Status: {'Success' if result['success'] else 'Failed'}\n"
            if result['success']:
                output += f"   Results: {result['num_results']}\n"
        
        return output