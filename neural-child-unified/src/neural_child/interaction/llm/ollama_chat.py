#----------------------------------------------------------------------------
#File:       ollama_chat.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Chat interface for neural child development using Ollama
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Chat interface for neural child development using Ollama.

Extracted from neural-child-init/ollama_chat.py
Configured for Ollama with gemma3:1b model and GPU support.
"""

import requests
import json
import torch
from typing import Dict, Any, Optional, Union
from datetime import datetime
import yaml
from pathlib import Path

# Import from unified llm_module
try:
    from neural_child.interaction.llm.llm_module import chat_completion
except ImportError:
    try:
        from llm_module import chat_completion
    except ImportError:
        chat_completion = None
        print("Warning: chat_completion not available. Some features may be limited.")

# Try to load config from config.yaml
_config = None
try:
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
except Exception as e:
    print(f"Warning: Could not load config.yaml: {str(e)}")

# Default configuration
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


class OllamaChat:
    """Interface for Ollama LLM integration"""
    
    def __init__(self, model: Optional[str] = None):
        """Initialize Ollama chat interface
        
        Args:
            model (str, optional): Model name (defaults to config or gemma3:1b)
        """
        # Get configuration values
        self.base_url = _get_config_value('ollama.base_url', DEFAULT_BASE_URL)
        default_model = _get_config_value('ollama.model', DEFAULT_MODEL)
        self.model = model if model is not None else default_model
        self.temperature = _get_config_value('ollama.temperature', DEFAULT_TEMPERATURE)
        self.max_tokens = _get_config_value('ollama.max_tokens', DEFAULT_MAX_TOKENS)
        self.timeout = _get_config_value('ollama.timeout', DEFAULT_TIMEOUT)
        self.context = []
        
    def generate_response(self, 
                         message: str, 
                         emotional_state: Dict[str, float],
                         brain_state: Dict[str, Any],
                         stage: str,
                         age_months: int,
                         image_data: Optional[Dict[str, str]] = None) -> str:
        """Generate response using Ollama with emotional context and optional image
        
        Args:
            message (str): User message
            emotional_state (Dict[str, float]): Current emotional state
            brain_state (Dict[str, Any]): Current brain state
            stage (str): Developmental stage
            age_months (int): Age in months
            image_data (Dict[str, str], optional): Image data for vision models
            
        Returns:
            str: Generated response
        """
        # Create system prompt with emotional and developmental context
        system_prompt = f"""You are a neural child AI at developmental stage {stage} (age {age_months} months).
Your current emotional state is:
{json.dumps(emotional_state, indent=2)}

Your brain state shows:
{json.dumps(brain_state, indent=2)}

Respond naturally as a child of this age and emotional state would.
Do not mention specific emotion values or numbers.
Express emotions through tone, word choice, and behavior.
Use *asterisks* to show physical reactions or behaviors.

Remember:
- Stay in character as a child AI
- Express emotions naturally through your response
- Be consistent with your developmental stage
- Show appropriate emotional reactions
- Address the user as Mr. Chris
"""

        try:
            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": message,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            # Add image data if provided and using vision model
            if image_data and ("llava" in self.model.lower() or "vision" in self.model.lower()):
                payload["images"] = [image_data.get("data", "")]
            
            # Make request to Ollama
            api_url = f"{self.base_url}/api/generate"
            response = requests.post(api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get('response', '')
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error generating Ollama response: {str(e)}")
            return "I'm having trouble processing that right now..."
            
    def get_sentiment(self, text: str) -> float:
        """Get sentiment score from Ollama
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        try:
            system_prompt = """Analyze the sentiment of the following text.
Return only a number between -1.0 (most negative) and 1.0 (most positive).
Do not include any other text or explanation."""

            payload = {
                "model": self.model,
                "prompt": text,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent sentiment analysis
                    "num_predict": 50  # Short response for sentiment
                }
            }
            
            api_url = f"{self.base_url}/api/generate"
            response = requests.post(api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            sentiment_text = result.get('response', '0.0')
            
            try:
                # Try to extract number from response
                import re
                numbers = re.findall(r'-?\d+\.?\d*', sentiment_text)
                if numbers:
                    sentiment = float(numbers[0])
                    return max(-1.0, min(1.0, sentiment))
                else:
                    return 0.0
            except ValueError:
                return 0.0
                
        except Exception as e:
            print(f"Error getting sentiment: {str(e)}")
            return 0.0


class OllamaChildChat:
    """Neural child chat system using Ollama"""
    
    def __init__(self, memory_store=None, emotional_system=None, language_system=None, model_name: Optional[str] = None):
        """Initialize neural child chat system
        
        Args:
            memory_store: Memory store instance (optional)
            emotional_system: Emotional system instance (optional)
            language_system: Language system instance (optional)
            model_name (str, optional): Model name (defaults to config or gemma3:1b)
        """
        self.ollama = OllamaChat(model=model_name)
        self.memory_store = memory_store
        self.emotional_system = emotional_system
        self.language_system = language_system
        self.last_interaction = None
        
    def chat(self, message: str) -> Dict[str, Any]:
        """Process chat message and generate response
        
        Args:
            message (str): User message
            
        Returns:
            Dict[str, Any]: Response with emotions, brain state, learning, and memory
        """
        try:
            # Update interaction time
            current_time = datetime.now()
            time_since_last = None
            if self.last_interaction:
                time_since_last = (current_time - self.last_interaction).total_seconds()
            self.last_interaction = current_time
            
            # Get current states (with fallbacks if systems not available)
            emotional_state = {
                'joy': 0.5,
                'trust': 0.5,
                'fear': 0.2,
                'surprise': 0.2,
                'sadness': 0.2,
                'anger': 0.2,
                'disgust': 0.2,
                'anticipation': 0.2
            }
            
            if self.emotional_system is not None and hasattr(self.emotional_system, 'emotional_state'):
                try:
                    if isinstance(self.emotional_system.emotional_state, torch.Tensor):
                        state = self.emotional_system.emotional_state
                        if len(state.shape) > 0 and state.shape[0] > 0:
                            emotional_state = {
                                'joy': float(state[0]) if len(state) > 0 else 0.5,
                                'trust': float(state[1]) if len(state) > 1 else 0.5,
                                'fear': float(state[2]) if len(state) > 2 else 0.2,
                                'surprise': float(state[3]) if len(state) > 3 else 0.2,
                                'sadness': float(state[4]) if len(state) > 4 else 0.2,
                                'anger': float(state[5]) if len(state) > 5 else 0.2,
                                'disgust': float(state[6]) if len(state) > 6 else 0.2,
                                'anticipation': float(state[7]) if len(state) > 7 else 0.2
                            }
                except Exception as e:
                    print(f"Warning: Error getting emotional state: {str(e)}")
            
            brain_state = {
                'arousal': 0.5,
                'attention': 0.5,
                'emotional_valence': 0.5,
                'consciousness': 0.5
            }
            
            if (self.emotional_system is not None and 
                hasattr(self.emotional_system, 'brain') and 
                hasattr(self.emotional_system.brain, 'brain_state')):
                try:
                    bs = self.emotional_system.brain.brain_state
                    brain_state = {
                        'arousal': float(bs.arousal) if hasattr(bs, 'arousal') else 0.5,
                        'attention': float(bs.attention) if hasattr(bs, 'attention') else 0.5,
                        'emotional_valence': float(bs.emotional_valence) if hasattr(bs, 'emotional_valence') else 0.5,
                        'consciousness': float(bs.consciousness) if hasattr(bs, 'consciousness') else 0.5
                    }
                except Exception as e:
                    print(f"Warning: Error getting brain state: {str(e)}")
            
            # Get stage and age (with fallbacks)
            stage = "NEWBORN"
            age_months = 0
            
            if (self.emotional_system is not None and 
                hasattr(self.emotional_system, 'curriculum') and
                hasattr(self.emotional_system.curriculum, 'current_stage')):
                try:
                    stage = self.emotional_system.curriculum.current_stage.name
                except Exception:
                    pass
            
            if self.emotional_system is not None and hasattr(self.emotional_system, 'age'):
                try:
                    age_months = self.emotional_system.age()
                except Exception:
                    pass
            
            # Generate response
            response = self.ollama.generate_response(
                message=message,
                emotional_state=emotional_state,
                brain_state=brain_state,
                stage=stage,
                age_months=age_months
            )
            
            # Get sentiment and update emotional state (if available)
            sentiment = self.ollama.get_sentiment(response)
            
            if self.emotional_system is not None and hasattr(self.emotional_system, 'update_emotions'):
                try:
                    # Create new emotional state tensor with all 8 emotions
                    new_emotional_state = torch.tensor([
                        max(0.1, emotional_state['joy'] + sentiment * 0.2),
                        max(0.1, emotional_state['trust'] + sentiment * 0.1),
                        max(0.1, emotional_state['fear'] - sentiment * 0.1),
                        emotional_state['surprise'],
                        max(0.1, emotional_state['sadness'] - sentiment * 0.15),
                        max(0.1, emotional_state['anger'] - sentiment * 0.1),
                        emotional_state['disgust'],
                        max(0.1, emotional_state['anticipation'] + sentiment * 0.15)
                    ])
                    
                    # Get device if available
                    if hasattr(self.emotional_system, 'emotional_state') and isinstance(self.emotional_system.emotional_state, torch.Tensor):
                        device = self.emotional_system.emotional_state.device
                        new_emotional_state = new_emotional_state.to(device)
                    
                    self.emotional_system.update_emotions(new_emotional_state)
                except Exception as e:
                    print(f"Warning: Error updating emotions: {str(e)}")
            
            # Process language learning (if available)
            learning = {}
            if self.language_system is not None and hasattr(self.language_system, 'process_input'):
                try:
                    learning = self.language_system.process_input(response)
                except Exception as e:
                    print(f"Warning: Error processing language learning: {str(e)}")
            
            # Create memory (if available)
            memory_id = None
            if self.memory_store is not None:
                try:
                    memory_type = "episodic"
                    memory_content = f"User: {message}\nResponse: {response}"
                    
                    # Create embedding for memory (if model available)
                    if hasattr(self.memory_store, 'model') and hasattr(self.memory_store.model, 'encode'):
                        embedding = self.memory_store.model.encode(memory_content).tolist()
                    else:
                        embedding = None
                    
                    if hasattr(self.memory_store, 'store_episodic_memory'):
                        memory_id = self.memory_store.store_episodic_memory(
                            content=memory_content,
                            embedding=embedding,
                            metadata={
                                'type': memory_type,
                                'sentiment': sentiment,
                                'time_since_last': time_since_last,
                                'stage': stage,
                                'age_months': age_months,
                                **emotional_state
                            }
                        )
                except Exception as e:
                    print(f"Warning: Error storing memory: {str(e)}")
            
            return {
                'response': response,
                'emotions': emotional_state,
                'brain_state': brain_state,
                'learning': learning,
                'memory': {
                    'type': 'episodic',
                    'content': f"User: {message}\nResponse: {response}",
                    'emotional_value': (sentiment + 1) / 2  # Convert to 0-1 range
                }
            }
            
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return {
                'response': "I'm having trouble processing that right now...",
                'emotions': emotional_state,
                'brain_state': brain_state
            }

    def process_emotional_input(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process emotional input and store in memory
        
        Args:
            input_text (str): Input text
            context (str, optional): Additional context
            
        Returns:
            Dict[str, Any]: Processing result with emotions and memory
        """
        try:
            # Get current emotional state (with fallbacks)
            emotional_state = {
                'joy': 0.5,
                'trust': 0.5,
                'fear': 0.2,
                'surprise': 0.2,
                'sadness': 0.2,
                'anger': 0.2,
                'disgust': 0.2,
                'anticipation': 0.2
            }
            
            if self.emotional_system is not None and hasattr(self.emotional_system, 'emotional_state'):
                try:
                    if isinstance(self.emotional_system.emotional_state, torch.Tensor):
                        state = self.emotional_system.emotional_state
                        if len(state.shape) > 0 and state.shape[0] > 0:
                            emotional_state = {
                                'joy': float(state[0]) if len(state) > 0 else 0.5,
                                'trust': float(state[1]) if len(state) > 1 else 0.5,
                                'fear': float(state[2]) if len(state) > 2 else 0.2,
                                'surprise': float(state[3]) if len(state) > 3 else 0.2,
                                'sadness': float(state[4]) if len(state) > 4 else 0.2,
                                'anger': float(state[5]) if len(state) > 5 else 0.2,
                                'disgust': float(state[6]) if len(state) > 6 else 0.2,
                                'anticipation': float(state[7]) if len(state) > 7 else 0.2
                            }
                except Exception as e:
                    print(f"Warning: Error getting emotional state: {str(e)}")
            
            # Calculate valence and arousal
            valence = (emotional_state['joy'] + emotional_state['trust'] + emotional_state['anticipation'] - 
                      emotional_state['fear'] - emotional_state['sadness'] - emotional_state['disgust']) / 3
            arousal = (emotional_state['surprise'] + emotional_state['fear'] + 
                      emotional_state['anger'] + emotional_state['anticipation']) / 4
            
            # Calculate intensity based on emotional extremes
            intensity = max(
                emotional_state['fear'],
                emotional_state['surprise'],
                emotional_state['anger'],
                emotional_state['joy'],
                abs(valence)
            )
            
            # Create memory (if available)
            memory_id = None
            if self.memory_store is not None and hasattr(self.memory_store, 'store_episodic_memory'):
                try:
                    memory_type = "episodic"
                    memory_content = f"User: {input_text}\nResponse: {intensity}"
                    memory_id = self.memory_store.store_episodic_memory(
                        content=memory_content,
                        emotional_state=emotional_state,
                        metadata={
                            'type': memory_type,
                            'intensity': intensity,
                            'context': context
                        }
                    )
                except Exception as e:
                    print(f"Warning: Error storing memory: {str(e)}")
            
            return {
                'response': intensity,
                'emotions': emotional_state,
                'memory': {
                    'type': 'episodic',
                    'content': memory_content if self.memory_store else f"User: {input_text}",
                    'emotional_value': (intensity + 1) / 2  # Convert to 0-1 range
                }
            }
            
        except Exception as e:
            print(f"Error processing emotional input: {str(e)}")
            return {
                'response': "I'm having trouble processing that right now...",
                'emotions': emotional_state
            }


def get_child_response(text: str, stage: str, age_months: int) -> str:
    """
    Get response from neural child based on developmental stage
    
    Args:
        text (str): Input text to respond to
        stage (str): Current developmental stage
        age_months (int): Age in months
        
    Returns:
        str: Generated response
    """
    try:
        prompt = f"""
        You are a neural child AI at developmental stage {stage} (age {age_months} months).
        Respond to the following input in a way appropriate for your developmental stage:
        
        Input: {text}
        
        Response:"""
        
        if chat_completion is not None:
            return chat_completion(prompt)
        else:
            return "I'm having trouble connecting to my language model."
        
    except Exception as e:
        print(f"Error getting child response: {e}")
        return ""


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, float]: Sentiment scores for different emotions
    """
    try:
        prompt = f"""
        Analyze the sentiment of the following text.
        Return scores (0-1) for: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
        
        Text: {text}
        
        Format response as JSON:
        {{
            "joy": 0.5,
            "trust": 0.3,
            "fear": 0.2,
            "surprise": 0.1,
            "sadness": 0.1,
            "disgust": 0.1,
            "anger": 0.1,
            "anticipation": 0.2
        }}
        """
        
        if chat_completion is not None:
            response = chat_completion(prompt, structured_output=True)
            return response if response else {}
        else:
            return {}
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {}

