"""
Hugging Face LLM Client Module
Handles communication with Hugging Face Router Chat Completions API for LLM inference
"""
import requests
import json
import time
import logging
import os
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceLLMClient:
    """Client for interacting with Hugging Face Router Chat Completions API"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "deepseek-ai/DeepSeek-V3.1-Terminus",
                 max_retries: int = 3):
        """
        Initialize Hugging Face LLM client
        
        Args:
            api_key: Hugging Face API key
            model: Model identifier on Hugging Face Hub
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY', '') or os.getenv('HF_TOKEN', '')
        self.model = model
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        self.max_retries = max_retries
        self.timeout = 30
    
    def create_headers(self) -> Dict[str, str]:
        """Create HTTP headers for API request"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """
        Call Hugging Face Chat Completions API with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        # Build messages array for chat completions
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
                    
                    # Parse chat completions response format
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            logger.debug(f"HF LLM call successful, received {len(content)} characters")
                            return content
                    
                    logger.warning(f"Unexpected HF response format: {result}")
                    return None
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                    logger.warning(f"HF Rate limit hit, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 503:  # Model loading
                    try:
                        error_data = response.json()
                        estimated_time = error_data.get('estimated_time', 20)
                    except:
                        estimated_time = 20
                    logger.info(f"Model loading, waiting {estimated_time}s")
                    time.sleep(estimated_time + 5)
                    continue
                
                else:
                    logger.error(f"HF API error: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"HF API timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"HF API error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        logger.error(f"Failed to get response from Hugging Face after {self.max_retries} attempts")
        return None

    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response (same interface as OpenRouter client)
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed JSON dict or None if extraction failed
        """
        if not response:
            return None
            
        # Try to find JSON blocks
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
            
        # Try to find JSON-like structures
        json_like_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_like_pattern, response)
        
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logger.warning("Could not extract JSON from HF LLM response")
        return None


# Available Hugging Face models for text generation
HF_TEXT_GENERATION_MODELS = [
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# Model descriptions for UI
HF_MODEL_DESCRIPTIONS = {
    "deepseek-ai/DeepSeek-V3.1-Terminus": "DeepSeek V3.1 Terminus - Latest high-performance model",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B - Fast and capable instruction model",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama 3.1 70B - High-quality large model",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3 - Excellent instruction following",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B - Mixture of experts model",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini - Lightweight but capable",
    "microsoft/Phi-3-medium-4k-instruct": "Phi-3 Medium - Balanced performance",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B - High quality instruction model",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B - Advanced Chinese/English model",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B - Larger Qwen model",
}