#----------------------------------------------------------------------------
#File:       theory_of_mind.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Theory of Mind system for understanding others' mental states
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Theory of Mind system for understanding others' mental states.

Extracted from neural-child-init/psychological_components.py
Merged with features from neural-child-1/theory_of_mind.py
Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional, Tuple

# Optional imports for unified structure
try:
    from neural_child.utils.logger import DevelopmentLogger
except ImportError:
    DevelopmentLogger = None


class TheoryOfMind(nn.Module):
    """Theory of Mind system for modeling others' mental states.
    
    Predicts emotional states, beliefs, intentions, and attention
    of other agents based on social context.
    """
    
    def __init__(self, device: Optional[torch.device] = None, input_dim: int = 398):
        """Initialize the Theory of Mind system.
        
        Args:
            device: Device to run on (defaults to cuda if available)
            input_dim: Dimension of social context input (default 398)
                Typically: base_dim (128) + sensory (256) + drives (10) + emotional (4) = 398
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        
        # Mental state predictor network
        self.mental_state_predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        # Perspective taking networks for different mental state aspects
        self.perspective_taking = nn.ModuleDict({
            'emotional': nn.Linear(128, 4),  # Emotional state prediction
            'belief': nn.Linear(128, 64),    # Belief state representation
            'intention': nn.Linear(128, 32), # Intention prediction
            'attention': nn.Linear(128, 16)  # Attention focus
        }).to(self.device)
        
        # Relationship memory for tracking interaction history
        self.relationship_memory = deque(maxlen=1000)
        
        # Social bias parameter (learnable, affects emotional predictions)
        self.social_bias = nn.Parameter(torch.ones(4, device=self.device))
        
    def forward(self, social_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict mental states from social context.
        
        Args:
            social_context: Social context input (shape: [batch, input_dim] or [input_dim])
            
        Returns:
            Dict with predictions for emotional, belief, intention, and attention
        """
        # Ensure input is on correct device
        if social_context.device != self.device:
            social_context = social_context.to(self.device)
        
        # Ensure correct shape (add batch dimension if needed)
        if social_context.dim() == 1:
            social_context = social_context.unsqueeze(0)
        
        # Predict mental state from social context
        mental_state = self.mental_state_predictor(social_context)
        
        # Generate predictions for different mental state aspects
        predictions = {
            'emotional': torch.sigmoid(self.perspective_taking['emotional'](mental_state)),
            'belief': torch.tanh(self.perspective_taking['belief'](mental_state)),
            'intention': torch.softmax(self.perspective_taking['intention'](mental_state), dim=-1),
            'attention': torch.sigmoid(self.perspective_taking['attention'](mental_state))
        }
        
        # Apply social bias to emotional predictions
        predictions['emotional'] = predictions['emotional'] * self.social_bias.unsqueeze(0)
        
        return predictions
    
    def update_relationship_model(self, interaction: torch.Tensor, outcome: float):
        """Update relationship model based on interaction outcome.
        
        Args:
            interaction: Interaction tensor (features of the interaction)
            outcome: Outcome value (positive = good, negative = bad)
        """
        # Ensure interaction is on correct device
        if interaction.device != self.device:
            interaction = interaction.to(self.device)
        
        # Store interaction in memory
        self.relationship_memory.append((interaction, outcome))
        
        # Update social bias based on recent outcomes (last 100 interactions)
        if len(self.relationship_memory) >= 100:
            recent_outcomes = torch.tensor(
                [o for _, o in list(self.relationship_memory)[-100:]],
                device=self.device
            )
            
            # Update social bias based on average outcome
            outcome_mean = recent_outcomes.mean()
            self.social_bias.data = torch.sigmoid(outcome_mean * self.social_bias.data)
    
    def get_social_bias(self) -> Dict[str, float]:
        """Get current social bias values.
        
        Returns:
            Dict with social bias for each emotional dimension
        """
        bias = self.social_bias.data.cpu().numpy()
        return {
            'joy': float(bias[0]) if len(bias) > 0 else 1.0,
            'trust': float(bias[1]) if len(bias) > 1 else 1.0,
            'fear': float(bias[2]) if len(bias) > 2 else 1.0,
            'surprise': float(bias[3]) if len(bias) > 3 else 1.0
        }
    
    def get_relationship_history_size(self) -> int:
        """Get number of interactions in relationship memory.
        
        Returns:
            Number of stored interactions
        """
        return len(self.relationship_memory)

