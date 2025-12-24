#----------------------------------------------------------------------------
#File:       attachment.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Attachment system for modeling caregiver-child relationships
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Attachment system for modeling caregiver-child relationships.

Extracted from neural-child-init/psychological_components.py
Merged with features from neural-child-1/attachment.py
Adapted imports to use unified structure.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional

# Optional imports for unified structure
try:
    from neural_child.utils.logger import DevelopmentLogger
except ImportError:
    DevelopmentLogger = None


class AttachmentSystem(nn.Module):
    """Attachment system modeling caregiver-child relationships.
    
    Models four attachment styles:
    - Secure (index 0): Healthy, trusting relationships
    - Anxious (index 1): Clingy, worried about abandonment
    - Avoidant (index 2): Distant, self-reliant
    - Disorganized (index 3): Inconsistent, confused patterns
    """
    
    def __init__(self, device: Optional[torch.device] = None, input_dim: int = 4):
        """Initialize the attachment system.
        
        Args:
            device: Device to run on (defaults to cuda if available)
            input_dim: Dimension of emotional input (default 4)
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        
        # Attachment style weightings (secure, anxious, avoidant, disorganized)
        # Start with secure attachment as default
        self.attachment_styles = nn.Parameter(
            torch.tensor([0.7, 0.1, 0.1, 0.1], device=self.device),
            requires_grad=True
        )
        
        # Trust network for evaluating caregiver reliability
        self.trust_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Bonding network for relationship features
        self.bonding_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64)
        ).to(self.device)
        
        # History of caregiving interactions
        self.caregiving_history = deque(maxlen=1000)
        
        # Trust level parameter (learnable)
        self.trust_level = nn.Parameter(torch.tensor(0.5, device=self.device))
        
    def forward(self, caregiver_input: torch.Tensor) -> Dict[str, Any]:
        """Process caregiver input and return attachment response.
        
        Args:
            caregiver_input: Emotional input from caregiver (shape: [batch, input_dim])
            
        Returns:
            Dict with trust_level, attachment_style, and bonding_features
        """
        # Ensure input is on correct device
        if caregiver_input.device != self.device:
            caregiver_input = caregiver_input.to(self.device)
        
        # Ensure correct shape
        if caregiver_input.dim() == 1:
            caregiver_input = caregiver_input.unsqueeze(0)
        
        # Predict trust from caregiver input
        trust_prediction = self.trust_network(caregiver_input)
        
        # Generate bonding features
        bonding_features = self.bonding_network(caregiver_input)
        
        # Update trust level with exponential moving average
        self.trust_level.data = 0.95 * self.trust_level.data + 0.05 * trust_prediction.mean()
        
        # Calculate attachment response based on styles and trust
        attachment_response = torch.softmax(self.attachment_styles * self.trust_level, dim=0)
        
        return {
            'trust_level': self.trust_level,
            'attachment_style': attachment_response,
            'bonding_features': bonding_features,
            'style_distribution': {
                'secure': float(attachment_response[0].item()),
                'anxious': float(attachment_response[1].item()),
                'avoidant': float(attachment_response[2].item()),
                'disorganized': float(attachment_response[3].item())
            }
        }
    
    def update_attachment(self, interaction_quality: float):
        """Update attachment styles based on interaction quality.
        
        Args:
            interaction_quality: Quality of interaction (0.0 to 1.0)
                - High (>0.8): Strengthens secure attachment
                - Low (<0.3): Strengthens disorganized attachment
        """
        # Clamp quality to valid range
        interaction_quality = max(0.0, min(1.0, interaction_quality))
        
        # Add to history
        self.caregiving_history.append(interaction_quality)
        
        # Update styles based on recent history (last 100 interactions)
        if len(self.caregiving_history) >= 100:
            recent_quality = torch.tensor(
                list(self.caregiving_history),
                device=self.device
            )
            quality_mean = recent_quality.mean().item()
            
            # High quality interactions strengthen secure attachment
            if quality_mean > 0.8:
                self.attachment_styles.data[0] *= 1.01
            # Low quality interactions strengthen disorganized attachment
            elif quality_mean < 0.3:
                self.attachment_styles.data[3] *= 1.01
            
            # Normalize to maintain probability distribution
            self.attachment_styles.data = torch.softmax(self.attachment_styles.data, dim=0)
    
    def get_attachment_style(self) -> Dict[str, float]:
        """Get current attachment style distribution.
        
        Returns:
            Dict with attachment style probabilities
        """
        styles = torch.softmax(self.attachment_styles.data, dim=0)
        return {
            'secure': float(styles[0].item()),
            'anxious': float(styles[1].item()),
            'avoidant': float(styles[2].item()),
            'disorganized': float(styles[3].item())
        }
    
    def get_trust_level(self) -> float:
        """Get current trust level.
        
        Returns:
            Trust level (0.0 to 1.0)
        """
        return float(self.trust_level.item())

