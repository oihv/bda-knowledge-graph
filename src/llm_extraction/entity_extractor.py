"""
Entity Extraction Module
Uses LLM to extract structured entities and relationships from text
"""
import json
import time
from typing import Dict, List, Optional
import logging
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from config import ENTITY_TYPES, RELATIONSHIP_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relationships from text using LLM"""
    
    def __init__(self, llm_client: LLMClient, batch_size: int = 10, delay_between_batches: float = 60.0):
        """
        Initialize entity extractor with rate limiting
        
        Args:
            llm_client: LLM client instance
            batch_size: Number of chunks to process before delay
            delay_between_batches: Seconds to wait between batches
        """
        self.llm_client = llm_client
        self.entity_types = ENTITY_TYPES
        self.relationship_types = RELATIONSHIP_TYPES
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
    
    def create_extraction_prompt(self, text: str) -> str:
        """
        Create prompt for entity and relationship extraction
        
        Args:
            text: Input text chunk
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are an expert at extracting structured information from financial and business documents.

Extract all entities and relationships from the following text. Focus on:

ENTITY TYPES:
{', '.join(self.entity_types)}

RELATIONSHIP TYPES:
{', '.join(self.relationship_types)}

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
    
    def extract_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract entities and relationships from a text chunk
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with entities and relationships or None
        """
        logger.info(f"Extracting entities from {len(text)} characters")
        
        prompt = self.create_extraction_prompt(text)
        system_prompt = "You are a precise information extraction system. Always respond with valid JSON."
        
        result = self.llm_client.call_llm_with_json_response(prompt, system_prompt)
        
        if result and self.validate_extraction(result):
            return result
        
        logger.warning("Failed to extract valid entities from text")
        return None
    
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
    
    def merge_extractions(self, extractions: List[Dict]) -> Dict:
        """
        Merge multiple extraction results, removing duplicates
        
        Args:
            extractions: List of extraction dictionaries
            
        Returns:
            Merged dictionary with unique entities and relationships
        """
        merged = {
            'entities': [],
            'relationships': []
        }
        
        entity_set = set()
        relationship_set = set()
        
        for extraction in extractions:
            if not extraction:
                continue
            
            # Merge entities
            for entity in extraction.get('entities', []):
                entity_key = (entity['name'].lower(), entity['type'])
                if entity_key not in entity_set:
                    entity_set.add(entity_key)
                    merged['entities'].append(entity)
            
            # Merge relationships
            for rel in extraction.get('relationships', []):
                rel_key = (
                    rel['source'].lower(),
                    rel['target'].lower(),
                    rel['type']
                )
                if rel_key not in relationship_set:
                    relationship_set.add(rel_key)
                    merged['relationships'].append(rel)
        
        logger.info(f"Merged to {len(merged['entities'])} unique entities "
                   f"and {len(merged['relationships'])} relationships")
        
        return merged
    
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