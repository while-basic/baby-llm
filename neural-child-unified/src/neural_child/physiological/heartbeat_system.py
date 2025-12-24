#----------------------------------------------------------------------------
#File:       heartbeat_system.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Heartbeat system for neural child development
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Heartbeat system for neural child development.

Extracted from neural-child-init/heartbeat_system.py
Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
from datetime import datetime, timedelta
import math
from typing import Dict, Optional, List, Any
from enum import Enum, auto
import numpy as np

# Optional imports for unified structure
try:
    from neural_child.utils.logger import DevelopmentLogger
except ImportError:
    DevelopmentLogger = None


class HeartRateState(Enum):
    """Heart rate states based on emotional and physiological conditions."""
    RESTING = auto()      # Normal resting state
    ELEVATED = auto()     # Increased due to positive excitement
    ANXIOUS = auto()      # Increased due to negative emotions
    FOCUSED = auto()      # Slightly elevated due to concentration
    RELAXED = auto()      # Lower than resting due to calm state


class HeartbeatSystem:
    """Heartbeat system simulating physiological heart rate responses.
    
    Models real-time heartbeat based on emotional state, with neural network
    modulation and memory-triggered responses.
    """
    
    def __init__(self, base_rate: int = 80, device: Optional[torch.device] = None):
        """Initialize the heartbeat system.
        
        Args:
            base_rate (int): Base heartbeat rate in beats per minute (default: 80)
            device: Device to run on (defaults to cuda if available)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core heartbeat parameters
        self.base_rate = base_rate
        self.current_rate = float(base_rate)
        self.last_update = datetime.now()
        
        # State tracking
        self.state = HeartRateState.RESTING
        self.state_history: List[Dict[str, Any]] = []
        
        # Emotional impact factors
        self.emotional_factors = {
            'joy': 1.2,      # Increases heart rate moderately
            'trust': 0.9,    # Slightly decreases heart rate
            'fear': 1.5,     # Significantly increases heart rate
            'surprise': 1.3,  # Temporarily increases heart rate
            'sadness': 0.95,  # Slightly decreases heart rate
            'anger': 1.4,    # Significantly increases heart rate
            'disgust': 1.1,  # Slightly increases heart rate
            'anticipation': 1.15  # Moderately increases heart rate
        }
        
        # Memory impact settings
        self.memory_impact_duration = timedelta(seconds=30)
        self.memory_decay_rate = 0.95
        
        # Rate limits
        self.min_rate = 60
        self.max_rate = 160
        
        # Initialize neural network for heart rate modulation
        self.rate_modulator = nn.Sequential(
            nn.Linear(len(self.emotional_factors), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(self.device)
        
    def update_from_emotional_state(self, emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Update heart rate based on emotional state.
        
        Args:
            emotional_state (Dict[str, float]): Current emotional state values
            
        Returns:
            Dict containing updated heart rate and state information
        """
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds()
        
        # Create emotional input tensor
        emotional_input = torch.tensor([
            emotional_state.get(emotion, 0.0) 
            for emotion in self.emotional_factors.keys()
        ], device=self.device).float().unsqueeze(0)
        
        # Get modulation factor from neural network
        with torch.no_grad():
            modulation = self.rate_modulator(emotional_input).item()
        
        # Calculate new heart rate
        emotional_impact = sum(
            emotional_state.get(emotion, 0.0) * factor
            for emotion, factor in self.emotional_factors.items()
        ) / len(self.emotional_factors)
        
        # Apply modulation and emotional impact
        target_rate = self.base_rate * emotional_impact * modulation
        
        # Smoothly transition to target rate
        rate_diff = target_rate - self.current_rate
        adjustment = rate_diff * min(1.0, time_delta / 5.0)  # 5-second full transition
        self.current_rate = max(self.min_rate, min(self.max_rate, 
                                                  self.current_rate + adjustment))
        
        # Update state
        self._update_state(emotional_state)
        
        # Record state history
        self.state_history.append({
            'timestamp': now.isoformat(),
            'rate': self.current_rate,
            'state': self.state.name,
            'emotional_state': emotional_state.copy(),
            'modulation': modulation,
            'emotional_impact': emotional_impact
        })
        
        # Trim history if too long
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        self.last_update = now
        
        return {
            'current_rate': self.current_rate,
            'state': self.state.name,
            'modulation': modulation,
            'emotional_impact': emotional_impact,
            'rate_change': adjustment
        }
        
    def process_memory_trigger(self, memory_valence: float, memory_intensity: float) -> Dict[str, Any]:
        """Process impact of memory recall on heart rate.
        
        Args:
            memory_valence (float): Emotional valence of memory (-1 to 1)
            memory_intensity (float): Intensity of memory (0 to 1)
            
        Returns:
            Dict containing memory impact information
        """
        # Clamp inputs to valid ranges
        memory_valence = max(-1.0, min(1.0, memory_valence))
        memory_intensity = max(0.0, min(1.0, memory_intensity))
        
        # Calculate memory impact
        impact_factor = memory_intensity * (1 + abs(memory_valence))
        rate_change = self.base_rate * 0.2 * impact_factor  # Max 20% change
        
        if memory_valence < 0:
            # Negative memories increase heart rate more
            rate_change *= 1.5
            
        # Apply change with decay
        self.current_rate = max(self.min_rate, min(self.max_rate,
                                                  self.current_rate + rate_change))
        
        return {
            'impact_factor': impact_factor,
            'rate_change': rate_change,
            'current_rate': self.current_rate
        }
    
    def _update_state(self, emotional_state: Dict[str, float]):
        """Update heart rate state based on emotional state."""
        # Calculate emotional factors
        anxiety = emotional_state.get('fear', 0.0) + emotional_state.get('anger', 0.0)
        joy = emotional_state.get('joy', 0.0) + emotional_state.get('anticipation', 0.0)
        calmness = emotional_state.get('trust', 0.0) - emotional_state.get('fear', 0.0)
        focus = emotional_state.get('attention', 0.0) if 'attention' in emotional_state else 0.0
        
        # Determine state
        if anxiety > 0.6:
            self.state = HeartRateState.ANXIOUS
        elif joy > 0.6:
            self.state = HeartRateState.ELEVATED
        elif focus > 0.7:
            self.state = HeartRateState.FOCUSED
        elif calmness > 0.5:
            self.state = HeartRateState.RELAXED
        else:
            self.state = HeartRateState.RESTING
            
    def get_current_heartbeat(self) -> Dict[str, Any]:
        """Get current heartbeat information.
        
        Returns:
            Dict with current rate, state, and timestamp
        """
        return {
            'rate': self.current_rate,
            'state': self.state.name,
            'last_update': self.last_update.isoformat(),
            'base_rate': self.base_rate
        }
        
    def get_heartbeat_history(self, 
                            start_time: Optional[datetime] = None, 
                            end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get heartbeat history within specified time range.
        
        Args:
            start_time: Start time for history (defaults to first entry)
            end_time: End time for history (defaults to now)
            
        Returns:
            List of heartbeat history entries
        """
        if not self.state_history:
            return []
            
        if not start_time:
            start_time = datetime.fromisoformat(self.state_history[0]['timestamp'])
        if not end_time:
            end_time = datetime.now()
            
        return [
            entry for entry in self.state_history
            if start_time <= datetime.fromisoformat(entry['timestamp']) <= end_time
        ]
    
    def get_rate_statistics(self) -> Dict[str, float]:
        """Get statistical information about heart rate history.
        
        Returns:
            Dict with mean, min, max, std deviation of heart rate
        """
        if not self.state_history:
            return {
                'mean': self.base_rate,
                'min': self.base_rate,
                'max': self.base_rate,
                'std': 0.0,
                'count': 0
            }
        
        rates = [entry['rate'] for entry in self.state_history]
        return {
            'mean': float(np.mean(rates)),
            'min': float(np.min(rates)),
            'max': float(np.max(rates)),
            'std': float(np.std(rates)),
            'count': len(rates)
        }

