"""
Entity Extraction Module
Uses LLM to extract structured entities and relationships from text
WITH: Validation, Normalization, and Metadata handling
"""
import json
import time
import re
from typing import Dict, List, Optional
import logging
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from config import ENTITY_TYPES, RELATIONSHIP_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NEW: Import validation and normalization classes
class EntityValidator:
    """Validate entities and relationships for logical consistency"""
    
    def __init__(self):
        # Define valid entity types
        self.valid_entity_types = {
            'Company', 'Organization', 'Person', 'Product', 
            'Location', 'Date', 'Money', 'Technology', 'Event'
        }
        
        # Define valid relationship schemas
        self.relationship_schemas = {
            'PRODUCES': {'source': {'Company', 'Organization'}, 'target': {'Product'}},
            'LOCATED_IN': {'source': {'Company', 'Person', 'Organization'}, 'target': {'Location'}},
            'EMPLOYS': {'source': {'Company', 'Organization'}, 'target': {'Person'}},
            'CEO_OF': {'source': {'Person'}, 'target': {'Company', 'Organization'}},
            'WORKS_FOR': {'source': {'Person'}, 'target': {'Company', 'Organization'}},
            'ACQUIRED': {'source': {'Company'}, 'target': {'Company', 'Product'}},
            'OWNS': {'source': {'Company', 'Person'}, 'target': {'Company', 'Product'}},
            'PARTNERS_WITH': {'source': {'Company'}, 'target': {'Company'}},
            'COMPETES_WITH': {'source': {'Company'}, 'target': {'Company'}},
            'INVESTS_IN': {'source': {'Company', 'Person'}, 'target': {'Company'}},
        }
        
        # Illogical relationship patterns to block
        self.forbidden_patterns = [
            ('Product', 'EMPLOYS', '*'),
            ('Location', 'PRODUCES', '*'),
            ('Date', '*', '*'),
            ('Money', '*', '*'),
        ]
    
    def validate_extraction(self, extraction: Dict) -> Dict:
        """Validate and clean an extraction"""
        valid_entities = self._validate_entities(extraction.get('entities', []))
        valid_relationships = self._validate_relationships(
            extraction.get('relationships', []),
            valid_entities
        )
        
        return {
            'entities': valid_entities,
            'relationships': valid_relationships
        }
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate entity structure and content"""
        valid = []
        
        for entity in entities:
            if not entity.get('name') or not entity.get('type'):
                continue
            
            name = entity['name'].strip()
            if len(name) < 2 or name.isdigit():
                continue
            
            # Update type if needed
            if entity['type'] not in self.valid_entity_types:
                entity['type'] = 'Organization'  # Default fallback
            
            valid.append(entity)
        
        return valid
    
    def _validate_relationships(self, relationships: List[Dict], 
                                entities: List[Dict]) -> List[Dict]:
        """Validate relationship logic and schema"""
        entity_lookup = {e['name']: e for e in entities}
        valid = []
        
        for rel in relationships:
            source_name = rel.get('source')
            target_name = rel.get('target')
            rel_type = rel.get('type')
            
            if not all([source_name, target_name, rel_type]):
                continue
            
            source_entity = entity_lookup.get(source_name)
            target_entity = entity_lookup.get(target_name)
            if not source_entity or not target_entity:
                continue
            
            # Check schema
            if not self._is_valid_relationship_schema(
                source_entity['type'], rel_type, target_entity['type']
            ):
                continue
            
            # Check forbidden patterns
            if self._is_forbidden_pattern(
                source_entity['type'], rel_type, target_entity['type']
            ):
                continue
            
            valid.append(rel)
        
        return valid
    
    def _is_valid_relationship_schema(self, source_type: str, 
                                     rel_type: str, target_type: str) -> bool:
        """Check if relationship matches defined schema"""
        if rel_type not in self.relationship_schemas:
            return True  # Allow unknown types
        
        schema = self.relationship_schemas[rel_type]
        return (source_type in schema['source'] and 
                target_type in schema['target'])
    
    def _is_forbidden_pattern(self, source_type: str, 
                             rel_type: str, target_type: str) -> bool:
        """Check if relationship matches forbidden pattern"""
        for pattern in self.forbidden_patterns:
            if pattern[1] == '*':
                if pattern[0] == source_type:
                    return True
            elif (pattern[0] == source_type and 
                  pattern[1] == rel_type and 
                  (pattern[2] == target_type or pattern[2] == '*')):
                return True
        return False


# NEW: Normalizer to handle duplicates
class EntityNormalizer:
    """Normalize entity names to prevent duplicates"""
    
    def __init__(self):
        self.remove_prefixes = [
            'the ', 'a ', 'an ', 'this ', 'that ',
            'the company ', 'the product ', 'the person '
        ]
        self.remove_suffixes = [
            ' inc', ' inc.', ' corp', ' corp.', ' ltd', ' ltd.',
            ' llc', ' co', ' co.'
        ]
        self.normalization_map = {}
        self.fuzzy_threshold = 0.90
    
    def normalize_name(self, name: str, entity_type: str = None) -> str:
        """Normalize a single entity name"""
        if not name:
            return ""
        
        normalized = name.strip()
        normalized_lower = normalized.lower()
        
        # Remove prefixes
        for prefix in self.remove_prefixes:
            if normalized_lower.startswith(prefix):
                normalized = normalized[len(prefix):]
                normalized_lower = normalized.lower()
        
        # Remove suffixes (for companies)
        if entity_type in ['Company', 'Organization']:
            for suffix in self.remove_suffixes:
                if normalized_lower.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
        
        # Clean and capitalize
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        if normalized and not normalized.isupper():
            words = normalized.split()
            normalized = ' '.join(w.capitalize() for w in words)
        
        return normalized
    
    def fuzzy_match(self, name1: str, name2: str) -> float:
        """Calculate similarity ratio"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def find_canonical_name(self, name: str, entity_type: str = None) -> str:
        """Find or create canonical name"""
        normalized = self.normalize_name(name, entity_type)
        
        if not normalized:
            return name
        
        # Check exact match
        if normalized in self.normalization_map:
            return normalized
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for canonical_name in self.normalization_map.keys():
            score = self.fuzzy_match(normalized, canonical_name)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = canonical_name
        
        if best_match:
            if normalized not in self.normalization_map[best_match]:
                self.normalization_map[best_match].append(normalized)
            return best_match
        
        # Create new
        self.normalization_map[normalized] = [normalized, name]
        return normalized
    
    def merge_duplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge duplicates based on normalization"""
        canonical_entities = {}
        
        for entity in entities:
            name = entity.get('name', '')
            entity_type = entity.get('type', '')
            
            canonical_name = self.find_canonical_name(name, entity_type)
            
            if canonical_name not in canonical_entities:
                canonical_entities[canonical_name] = {
                    'name': canonical_name,
                    'type': entity_type,
                    'original_names': [name],
                    'properties': entity.get('properties', {}).copy()
                }
            else:
                existing = canonical_entities[canonical_name]
                if name not in existing['original_names']:
                    existing['original_names'].append(name)
                # Merge properties (prefer non-empty)
                for key, value in entity.get('properties', {}).items():
                    if value and not existing['properties'].get(key):
                        existing['properties'][key] = value
        
        return list(canonical_entities.values())


class EntityExtractor:
    """Extract entities and relationships from text using LLM"""
    
    def __init__(self, llm_client: LLMClient, batch_size: int = 10, 
                 delay_between_batches: float = 60.0, use_validation: bool = True):
        """
        Initialize entity extractor with validation and normalization
        
        Args:
            llm_client: LLM client instance
            batch_size: Number of chunks to process before delay
            delay_between_batches: Seconds to wait between batches
            use_validation: Enable validation and filtering
        """
        self.llm_client = llm_client
        self.entity_types = ENTITY_TYPES
        self.relationship_types = RELATIONSHIP_TYPES
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.use_validation = use_validation
        
        # NEW: Initialize validator and normalizer
        self.validator = EntityValidator() if use_validation else None
        self.normalizer = EntityNormalizer()
        
        # NEW: Non-entity patterns to filter out
        self.non_entity_patterns = [
            r'^(revenue|profit|loss|income|expense|cost|sales)$',
            r'^(increase|decrease|growth|decline)$',
            r'^(report|document|statement|analysis)$',
            r'^(quarter|year|month|period)$',
            r'^\d+$',  # Pure numbers
        ]
    
    def create_extraction_prompt(self, text: str, metadata: Dict = None) -> str:
        """
        Create prompt for entity and relationship extraction
        NOW INCLUDES: Document context from metadata
        
        Args:
            text: Input text chunk
            metadata: Optional chunk metadata
            
        Returns:
            Formatted prompt
        """
        # NEW: Add document context
        doc_context = ""
        if metadata:
            doc_context = f"\n[Document Context: {metadata.get('document_name', 'Unknown')}"
            if metadata.get('document_type'):
                doc_context += f" | Type: {metadata['document_type']}"
            if metadata.get('sections_included'):
                doc_context += f" | Sections: {', '.join(metadata['sections_included'][:2])}"
            doc_context += "]\n"
        
        prompt = f"""You are an expert financial analyst and knowledge graph builder. Therefore, you are also an expert at extracting unstructured information from financial and business documents, company annual reports, financial filings, and business news.
{doc_context}
Extract all entities and relationships from the following text. Focus on:

ENTITY TYPES:
{', '.join(self.entity_types)}

RELATIONSHIP TYPES:
{', '.join(self.relationship_types)}

IMPORTANT RULES:
1. Only extract CONCRETE entities (companies, products, people, locations)
2. Do NOT extract abstract concepts like "revenue", "growth", "profit"
3. Do NOT extract generic terms like "the company", "the product"
4. Use FULL proper names (e.g., "Apple Inc." not "the company")
5. Ensure relationships are logically valid

TEXT TO ANALYZE:
{text}

Return your response in the following JSON format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "EntityType",
      "properties": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{"key": "value"}}
    }}
  ]
}}

Guidelines:
- Extract specific company names, people, products, locations, etc.
- Identify clear relationships between entities
- Include relevant properties (e.g., amounts, dates, percentages)
- Use exact entity names as they appear in the text
- Only include relationships explicitly stated or strongly implied

Respond with ONLY the JSON, no additional text."""
        
        return prompt
    
    def extract_from_text(self, chunk) -> Optional[Dict]:
        """
        Extract entities and relationships from a text chunk
        NOW HANDLES: Both string chunks and dict chunks with metadata
        
        Args:
            chunk: Text string OR dictionary with 'text' and metadata
            
        Returns:
            Dictionary with entities and relationships or None
        """
        # NEW: Handle both formats
        if isinstance(chunk, dict):
            text = chunk.get('text', '')
            metadata = chunk
        else:
            text = chunk
            metadata = {}
        
        logger.info(f"Extracting entities from {len(text)} characters")
        
        # NEW: Pass metadata to prompt
        prompt = self.create_extraction_prompt(text, metadata)
        system_prompt = "You are a precise information extraction system. Always respond with valid JSON."
        
        result = self.llm_client.call_llm_with_json_response(prompt, system_prompt)
        
        if result and self.validate_extraction(result):
            # NEW: Add document metadata to entities
            if metadata:
                for entity in result.get('entities', []):
                    entity['source_document'] = metadata.get('document_name')
                    entity['source_type'] = metadata.get('document_type')
            
            # NEW: Filter and validate
            if self.use_validation:
                result = self._filter_non_entities(result)
                result = self.validator.validate_extraction(result)
            
            return result
        
        logger.warning("Failed to extract valid entities from text")
        return None
    
    # NEW: Filter out non-entities
    def _filter_non_entities(self, extraction: Dict) -> Dict:
        """Remove abstract concepts and non-entities"""
        filtered_entities = []
        
        for entity in extraction.get('entities', []):
            name = entity['name'].lower()
            
            # Check against non-entity patterns
            is_non_entity = False
            for pattern in self.non_entity_patterns:
                if re.match(pattern, name):
                    is_non_entity = True
                    break
            
            # Check for generic references
            if name in ['the company', 'the product', 'the person', 'it', 'they']:
                is_non_entity = True
            
            if not is_non_entity:
                filtered_entities.append(entity)
        
        extraction['entities'] = filtered_entities
        
        # Filter relationships to only include valid entities
        valid_names = {e['name'] for e in filtered_entities}
        extraction['relationships'] = [
            rel for rel in extraction.get('relationships', [])
            if rel['source'] in valid_names and rel['target'] in valid_names
        ]
        
        return extraction
    
    def validate_extraction(self, data: Dict) -> bool:
        """
        Validate extracted data structure
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        if 'entities' not in data or 'relationships' not in data:
            return False
        
        if not isinstance(data['entities'], list) or not isinstance(data['relationships'], list):
            return False
        
        # Validate entity structure
        for entity in data['entities']:
            if not isinstance(entity, dict):
                return False
            if 'name' not in entity or 'type' not in entity:
                return False
        
        # Validate relationship structure
        for rel in data['relationships']:
            if not isinstance(rel, dict):
                return False
            if 'source' not in rel or 'target' not in rel or 'type' not in rel:
                return False
        
        return True
    
    # NEW: Helper to get entity type
    def _get_entity_type(self, name: str, entities: List[Dict]) -> Optional[str]:
        """Get entity type by name"""
        for entity in entities:
            if entity['name'] == name or name in entity.get('original_names', []):
                return entity['type']
        return None
    
    # NEW: Deduplicate relationships
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (rel['source'], rel['type'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def merge_extractions(self, extractions: List[Dict]) -> Dict:
        """
        Merge multiple extraction results with normalization and deduplication
        NOW INCLUDES: Normalization, deduplication, and canonical naming
        
        Args:
            extractions: List of extraction dictionaries
            
        Returns:
            Merged dictionary with unique entities and relationships
        """
        all_entities = []
        all_relationships = []
        
        for extraction in extractions:
            if not extraction:
                continue
            all_entities.extend(extraction.get('entities', []))
            all_relationships.extend(extraction.get('relationships', []))
        
        # NEW: Normalize and merge entities
        merged_entities = self.normalizer.merge_duplicate_entities(all_entities)
        
        # NEW: Update relationship references to use canonical names
        normalized_relationships = []
        
        for rel in all_relationships:
            source_canonical = self.normalizer.find_canonical_name(
                rel['source'], 
                self._get_entity_type(rel['source'], merged_entities)
            )
            target_canonical = self.normalizer.find_canonical_name(
                rel['target'],
                self._get_entity_type(rel['target'], merged_entities)
            )
            
            normalized_relationships.append({
                'source': source_canonical,
                'type': rel['type'],
                'target': target_canonical,
                'properties': rel.get('properties', {})
            })
        
        # NEW: Deduplicate relationships
        unique_rels = self._deduplicate_relationships(normalized_relationships)
        
        logger.info(f"Merged to {len(merged_entities)} unique entities "
                   f"and {len(unique_rels)} relationships")
        
        return {
            'entities': merged_entities,
            'relationships': unique_rels
        }
    
    def extract_from_document(self, document: Dict) -> Dict:
        """
        Extract entities and relationships from entire document
        
        Args:
            document: Preprocessed document with chunks
            
        Returns:
            Merged extraction results
        """
        logger.info(f"Processing document: {document.get('filename', 'Unknown')}")
        
        chunks = document.get('chunks', [])
        if not chunks:
            logger.warning("No chunks found in document")
            return {'entities': [], 'relationships': []}
        
        extractions = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            
            # Rate limiting: pause between batches
            if i > 0 and i % self.batch_size == 0:
                logger.info(f"Completed batch of {self.batch_size} chunks. Pausing for {self.delay_between_batches}s to avoid rate limits...")
                time.sleep(self.delay_between_batches)
            
            extraction = self.extract_from_text(chunk)
            if extraction:
                extractions.append(extraction)
                logger.info(f"Extracted {len(extraction.get('entities', []))} entities and {len(extraction.get('relationships', []))} relationships")
            
            # Longer delay between individual requests for free tier
            if i < len(chunks) - 1:  # Don't delay after the last chunk
                time.sleep(10.0)  # 10 second delay between individual requests
        
        merged = self.merge_extractions(extractions)
        
        # Add document context to result
        merged['document_name'] = document.get('filename', 'Unknown')
        merged['source_file'] = document.get('filepath', '')
        
        return merged
    
    def extract_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract from multiple documents
        
        Args:
            documents: List of preprocessed documents
            
        Returns:
            List of extraction results
        """
        results = []
        
        for doc in documents:
            result = self.extract_from_document(doc)
            results.append(result)
        
        return results
    
    def create_global_extraction(self, document_extractions: List[Dict]) -> Dict:
        """
        Create a single merged extraction from all documents
        
        Args:
            document_extractions: List of document extraction results
            
        Returns:
            Global merged extraction
        """
        logger.info(f"Creating global extraction from {len(document_extractions)} documents")
        
        global_extraction = self.merge_extractions(document_extractions)
        
        # Add metadata
        global_extraction['num_documents'] = len(document_extractions)
        global_extraction['document_sources'] = [
            ext.get('document_name', 'Unknown') 
            for ext in document_extractions
        ]
        
        return global_extraction