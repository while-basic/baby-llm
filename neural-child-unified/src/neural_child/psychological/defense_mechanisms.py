#----------------------------------------------------------------------------
#File:       defense_mechanisms.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Defense mechanisms system for coping with anxiety and stress
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Defense mechanisms system for coping with anxiety and stress.

Extracted from neural-child-init/psychological_components.py
Merged with features from neural-child-1/defense_mechanisms.py
Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

# Optional imports for unified structure
try:
    from neural_child.utils.logger import DevelopmentLogger
except ImportError:
    DevelopmentLogger = None


class DefenseMechanisms(nn.Module):
    """Defense mechanisms system for coping with anxiety and stress.
    
    Implements seven defense mechanisms:
    - Repression: Unconscious blocking of thoughts/feelings
    - Projection: Attributing own feelings to others
    - Denial: Refusing to acknowledge reality
    - Sublimation: Channeling negative energy into positive actions
    - Rationalization: Creating logical explanations for behaviors
    - Displacement: Redirecting emotions to safer targets
    - Regression: Reverting to earlier developmental stage
    """
    
    def __init__(self, device: Optional[torch.device] = None, input_dim: int = 398):
        """Initialize the defense mechanisms system.
        
        Args:
            device: Device to run on (defaults to cuda if available)
            input_dim: Dimension of emotional input (default 398)
                Typically: base_dim (128) + sensory (256) + drives (10) + emotional (4) = 398
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        
        # Network for determining mechanism strength
        self.mechanism_strength = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        ).to(self.device)
        
        # Individual defense mechanism networks
        self.mechanisms = nn.ModuleDict({
            'repression': nn.Linear(128, 1),
            'projection': nn.Linear(128, 1),
            'denial': nn.Linear(128, 1),
            'sublimation': nn.Linear(128, 1),
            'rationalization': nn.Linear(128, 1),
            'displacement': nn.Linear(128, 1),
            'regression': nn.Linear(128, 1)
        }).to(self.device)
        
        # Anxiety threshold (learnable parameter)
        # Defenses activate when anxiety exceeds this threshold
        self.anxiety_threshold = nn.Parameter(torch.tensor(0.7, device=self.device))
        
    def forward(self, emotional_input: torch.Tensor, anxiety_level: torch.Tensor) -> Dict[str, Any]:
        """Process emotional input and return defense mechanism activations.
        
        Args:
            emotional_input: Emotional state input (shape: [batch, input_dim] or [input_dim])
            anxiety_level: Current anxiety level (scalar or tensor)
            
        Returns:
            Dict with active_defense, defense_strength, and all_mechanisms
        """
        # Ensure inputs are on correct device
        if emotional_input.device != self.device:
            emotional_input = emotional_input.to(self.device)
        if isinstance(anxiety_level, torch.Tensor) and anxiety_level.device != self.device:
            anxiety_level = anxiety_level.to(self.device)
        
        # Ensure correct shape
        if emotional_input.dim() == 1:
            emotional_input = emotional_input.unsqueeze(0)
        
        # Convert anxiety_level to tensor if needed
        if not isinstance(anxiety_level, torch.Tensor):
            anxiety_level = torch.tensor(anxiety_level, device=self.device)
        
        # Check if anxiety exceeds threshold
        if anxiety_level.item() > self.anxiety_threshold.item():
            # Compute mechanism features
            mechanism_features = self.mechanism_strength(emotional_input)
            
            # Compute activations for all mechanisms
            defense_activations = {
                name: torch.sigmoid(layer(mechanism_features))
                for name, layer in self.mechanisms.items()
            }
            
            # Find strongest defense mechanism
            strongest_defense = max(
                defense_activations.items(),
                key=lambda x: x[1].item()
            )
            
            return {
                'active_defense': strongest_defense[0],
                'defense_strength': strongest_defense[1],
                'all_mechanisms': defense_activations,
                'anxiety_level': anxiety_level,
                'threshold': self.anxiety_threshold
            }
        
        # No defense activated if anxiety is below threshold
        return {
            'active_defense': None,
            'defense_strength': torch.tensor(0.0, device=self.device),
            'all_mechanisms': {
                name: torch.tensor(0.0, device=self.device)
                for name in self.mechanisms.keys()
            },
            'anxiety_level': anxiety_level,
            'threshold': self.anxiety_threshold
        }
    
    def update_threshold(self, stress_level: float):
        """Update anxiety threshold based on stress level.
        
        Args:
            stress_level: Current stress level (0.0 to 1.0)
                - Higher stress increases threshold (less sensitive)
                - Lower stress decreases threshold (more sensitive)
        """
        # Clamp stress level to valid range
        stress_level = max(0.0, min(1.0, stress_level))
        
        # Adjust threshold based on stress
        # Higher stress -> higher threshold (adaptation)
        # Lower stress -> lower threshold (sensitivity)
        adjustment = (stress_level - 0.5) * 0.1
        new_threshold = self.anxiety_threshold.data * (1.0 + adjustment)
        
        # Clamp threshold to reasonable range
        self.anxiety_threshold.data = torch.clamp(
            new_threshold,
            min=0.3,
            max=0.9
        )
    
    def get_anxiety_threshold(self) -> float:
        """Get current anxiety threshold.
        
        Returns:
            Anxiety threshold value (0.0 to 1.0)
        """
        return float(self.anxiety_threshold.item())
    
    def get_available_mechanisms(self) -> List[str]:
        """Get list of available defense mechanisms.
        
        Returns:
            List of mechanism names
        """
        return list(self.mechanisms.keys())

