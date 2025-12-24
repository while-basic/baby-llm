#----------------------------------------------------------------------------
#File:       helpers.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Helper utility functions
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Helper utility functions.

Merged from:
- neural-child-1/utils.py
- neural-child-init utilities
"""

import json
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime


def parse_llm_response(response_text: str, default_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse LLM response and ensure it matches expected format.

    Args:
        response_text: Raw response text from LLM
        default_response: Default response to use if parsing fails

    Returns:
        Parsed response dictionary
    """
    if default_response is None:
        default_response = {
            'content': 'I understand.',
            'emotional_context': {
                'joy': 0.5,
                'trust': 0.5,
                'fear': 0.1,
                'surprise': 0.3
            }
        }

    try:
        if isinstance(response_text, dict):
            return response_text

        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            try:
                parsed = json.loads(json_str)
                # Ensure required fields exist
                if 'content' not in parsed:
                    parsed['content'] = default_response['content']
                if 'emotional_context' not in parsed:
                    parsed['emotional_context'] = default_response['emotional_context']
                return parsed
            except json.JSONDecodeError:
                return default_response
    except Exception:
        return default_response


def ensure_tensor(
    data: Union[List, np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert input data to tensor and ensure it's on the correct device.

    Args:
        data: Input data (list, numpy array, or tensor)
        device: Target device (if None, uses CUDA if available)

    Returns:
        Tensor on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, list):
        return torch.tensor(data, device=device)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def format_time_delta(start_time: datetime) -> str:
    """Format time delta in a human-readable format.

    Args:
        start_time: Start time

    Returns:
        Formatted time delta string
    """
    delta = datetime.now() - start_time
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = delta.seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


def calculate_moving_average(values: List[float], window: int = 10) -> List[float]:
    """Calculate moving average with specified window size.

    Args:
        values: List of values
        window: Window size for moving average

    Returns:
        List of moving average values
    """
    if not values:
        return []
    if len(values) < window:
        return values

    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        result.append(sum(window_values) / len(window_values))
    return result


def extract_action_marker(text: str) -> Optional[str]:
    """Extract action marker from text enclosed in [brackets].

    Args:
        text: Text to search

    Returns:
        Action marker if found, None otherwise
    """
    try:
        start = text.find('[')
        end = text.find(']')
        if start != -1 and end != -1 and end > start:
            return text[start + 1:end].strip()
        return None
    except Exception:
        return None


def validate_emotional_vector(
    vector: Union[List, np.ndarray, torch.Tensor],
    emotion_dim: int = 4
) -> bool:
    """Validate that emotional vector values are within valid range.

    Args:
        vector: Emotional vector to validate
        emotion_dim: Expected dimension

    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()
        elif isinstance(vector, list):
            vector = np.array(vector)

        return (
            len(vector) == emotion_dim and
            np.all(vector >= 0) and
            np.all(vector <= 1)
        )
    except Exception:
        return False


def create_error_response(
    error_type: str,
    details: str,
    default_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a standardized error response.

    Args:
        error_type: Type of error
        details: Error details
        default_response: Default response to include

    Returns:
        Error response dictionary
    """
    if default_response is None:
        default_response = {
            'content': 'I encountered an error.',
            'emotional_context': {
                'joy': 0.2,
                'trust': 0.3,
                'fear': 0.4,
                'surprise': 0.3
            }
        }

    return {
        'error': True,
        'error_type': error_type,
        'details': details,
        'timestamp': datetime.now().isoformat(),
        'response': default_response
    }

