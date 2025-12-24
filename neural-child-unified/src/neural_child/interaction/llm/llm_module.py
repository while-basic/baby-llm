#----------------------------------------------------------------------------
#File:       llm_module.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: LLM integration for neural child development using Ollama
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""LLM integration module for neural child development.

Extracted from neural-child-init/llm_module.py
Merged with features from neural-child-4/neuralchild/utils/llm_module.py
Configured for Ollama with gemma3:1b model and GPU support.
"""

import requests
from typing import Dict, Any, Optional, Union
import json
import time
import random
import yaml
from pathlib import Path

# Try to load config from config.yaml
_config = None
try:
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
except Exception as e:
    print(f"Warning: Could not load config.yaml: {str(e)}")

# Default configuration (will be overridden by config.yaml if available)
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120

def _get_config_value(key_path: str, default):
    """Get configuration value from nested config dict"""
    if _config is None:
        return default
    
    keys = key_path.split('.')
    value = _config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def chat_completion(
    prompt: str, 
    structured_output: bool = False,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry_count: int = 3,
    retry_delay: float = 2.0
) -> Union[str, Dict[str, Any]]:
    """
    Get completion from local Ollama instance
    
    Args:
        prompt (str): The prompt to send to the model
        structured_output (bool): Whether to parse response as JSON
        model (str, optional): Model name (defaults to config or gemma3:1b)
        temperature (float, optional): Temperature setting (defaults to config or 0.7)
        max_tokens (int, optional): Maximum tokens to generate (defaults to config or 4096)
        retry_count (int): Number of retries on failure
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        Union[str, Dict[str, Any]]: Model response as string or parsed JSON
    """
    # Get configuration values
    base_url = _get_config_value('ollama.base_url', DEFAULT_BASE_URL)
    default_model = _get_config_value('ollama.model', DEFAULT_MODEL)
    default_temperature = _get_config_value('ollama.temperature', DEFAULT_TEMPERATURE)
    default_max_tokens = _get_config_value('ollama.max_tokens', DEFAULT_MAX_TOKENS)
    timeout = _get_config_value('ollama.timeout', DEFAULT_TIMEOUT)
    
    # Use provided values or defaults
    model_name = model if model is not None else default_model
    temp = temperature if temperature is not None else default_temperature
    max_toks = max_tokens if max_tokens is not None else default_max_tokens
    
    # Add structured output instructions if needed
    system_prompt = ""
    if structured_output:
        system_prompt = "\n\nProvide your response in JSON format. Return only valid JSON, no additional text."
    
    # Attempt the API request with retries
    for attempt in range(retry_count):
        try:
            # Prepare the request payload
            payload = {
                'model': model_name,
                'prompt': prompt + system_prompt,
                'stream': False,
                'options': {
                    'temperature': temp,
                    'num_predict': max_toks
                }
            }
            
            # Make request to Ollama
            api_url = f"{base_url}/api/generate"
            response = requests.post(
                api_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                
                if structured_output:
                    try:
                        # Try to parse as JSON
                        # Handle cases where JSON might be wrapped in ```json blocks
                        if "```json" in text:
                            json_start = text.find("```json") + 7
                            json_end = text.find("```", json_start)
                            if json_end == -1:
                                json_end = len(text)
                            json_content = text[json_start:json_end].strip()
                        elif "```" in text:
                            json_start = text.find("```") + 3
                            json_end = text.find("```", json_start)
                            if json_end == -1:
                                json_end = len(text)
                            json_content = text[json_start:json_end].strip()
                        else:
                            json_content = text.strip()
                        
                        # Parse the JSON
                        parsed = json.loads(json_content)
                        return parsed
                    except json.JSONDecodeError as e:
                        # If all retries fail to parse JSON, return empty dict or raw text
                        if attempt < retry_count - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"Warning: Could not parse JSON response: {str(e)}")
                            return {}
                
                return text.strip()
            else:
                # Non-200 status code
                error_msg = f"Ollama API returned status {response.status_code}"
                if attempt < retry_count - 1:
                    print(f"Warning: {error_msg}, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Error: {error_msg}")
                    return "" if structured_output else "I apologize, but I'm having trouble connecting to my language model."
                    
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API request failed (attempt {attempt+1}/{retry_count}): {str(e)}"
            if attempt < retry_count - 1:
                print(f"Warning: {error_msg}, retrying...")
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.5, 1.5)
                time.sleep(retry_delay * jitter)
                continue
            else:
                print(f"Error: {error_msg}")
                return {} if structured_output else "I apologize, but I'm having trouble connecting to my language model."
        
        except Exception as e:
            error_msg = f"Unexpected error in chat completion (attempt {attempt+1}/{retry_count}): {str(e)}"
            if attempt < retry_count - 1:
                print(f"Warning: {error_msg}, retrying...")
                jitter = random.uniform(0.5, 1.5)
                time.sleep(retry_delay * jitter)
                continue
            else:
                print(f"Error: {error_msg}")
                return {} if structured_output else "I apologize, but I'm having trouble connecting to my language model."
    
    # If we get here, all retries failed
    return {} if structured_output else "I apologize, but I'm having trouble connecting to my language model."

