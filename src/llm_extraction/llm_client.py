"""
LLM Client Module
Handles communication with OpenRouter API for LLM inference
"""
import requests
import json
import time
import logging
import re
from typing import Dict, Optional, List
from config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    LLM_MODEL,
    MAX_RETRIES,
    TIMEOUT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with OpenRouter LLM API"""
    
    def __init__(self, api_key: str = OPENROUTER_API_KEY, 
                 model: str = LLM_MODEL,
                 max_retries: int = MAX_RETRIES):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model = model
        self.base_url = OPENROUTER_BASE_URL
        self.max_retries = max_retries
        self.timeout = TIMEOUT
    
    def create_headers(self) -> Dict[str, str]:
        """Create HTTP headers for API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Financial Knowledge Graph"
        }
    
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """
        Call LLM API with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.create_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.debug(f"LLM call successful, received {len(content)} characters")
                    return content
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
            
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def call_llm_with_json_response(self, prompt: str, 
                                    system_prompt: Optional[str] = None) -> Optional[Dict]:
        """
        Call LLM and parse JSON response with better error handling
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt
            
        Returns:
            Parsed JSON dictionary or None
        """
        response_text = self.call_llm(prompt, system_prompt, temperature=0.2, max_tokens=3000)
        
        if not response_text:
            return None
        
        # Try to extract and parse JSON from response
        try:
            # Remove markdown code blocks
            cleaned = response_text
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            # Try direct parsing
            return json.loads(cleaned)
        
        except json.JSONDecodeError:
            # Try to find JSON object in text
            try:
                # Look for {..."entities": [...]} pattern
                match = re.search(r'\{[\s\S]*"entities"[\s\S]*"relationships"[\s\S]*\}', response_text)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)
            except:
                pass
            
            # Try to repair common JSON issues
            try:
                # Fix unterminated strings by finding last complete entity/relationship
                repaired = self._repair_json(response_text)
                if repaired:
                    return json.loads(repaired)
            except:
                pass
            
            logger.error(f"Failed to parse JSON response")
            logger.debug(f"Response text: {response_text[:500]}...")
            return None
    
    def _repair_json(self, text: str) -> Optional[str]:
        """Attempt to repair malformed JSON"""
        try:
            # Find the last complete entity
            entities_match = re.findall(r'"entities":\s*\[(.*?)\]', text, re.DOTALL)
            relationships_match = re.findall(r'"relationships":\s*\[(.*?)\]', text, re.DOTALL)
            
            if entities_match or relationships_match:
                entities = entities_match[0] if entities_match else "[]"
                relationships = relationships_match[0] if relationships_match else "[]"
                
                # Reconstruct JSON
                repaired = f'{{"entities": [{entities}], "relationships": [{relationships}]}}'
                return repaired
        except:
            pass
        
        return None
    
    def batch_process(self, prompts: List[str], system_prompt: Optional[str] = None,
                     delay: float = 1.0) -> List[Optional[str]]:
        """
        Process multiple prompts with delay between calls
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt for all calls
            delay: Delay in seconds between calls
            
        Returns:
            List of responses
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i + 1}/{len(prompts)}")
            response = self.call_llm(prompt, system_prompt)
            results.append(response)
            
            # Add delay between calls to avoid rate limits
            if i < len(prompts) - 1:
                time.sleep(delay)
        
        return results


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API key
    Returns pre-defined responses based on prompt patterns
    """
    
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """
        Return mock response based on prompt content
        """
        logger.info("Using Mock LLM Client (no API calls)")
        
        # Simulate processing delay
        time.sleep(0.5)
        
        # Check if this is an entity extraction request
        if "extract entities" in prompt.lower() or "json format" in prompt.lower():
            return self._mock_entity_extraction()
        
        # Default mock response
        return "This is a mock response from the LLM client."
    
    def _mock_entity_extraction(self) -> str:
        """Return mock entity extraction JSON"""
        mock_data = {
            "entities": [
                {
                    "name": "Samsung Electronics",
                    "type": "Company",
                    "properties": {"industry": "Technology", "location": "Seoul"}
                },
                {
                    "name": "Jong-Hee Han",
                    "type": "Person",
                    "properties": {"role": "CEO"}
                },
                {
                    "name": "Galaxy AI",
                    "type": "Product",
                    "properties": {"category": "AI Platform"}
                }
            ],
            "relationships": [
                {
                    "source": "Jong-Hee Han",
                    "target": "Samsung Electronics",
                    "type": "CEO_OF",
                    "properties": {}
                },
                {
                    "source": "Samsung Electronics",
                    "target": "Galaxy AI",
                    "type": "DEVELOPS",
                    "properties": {}
                }
            ]
        }
        
        return f"```json\n{json.dumps(mock_data, indent=2)}\n```"