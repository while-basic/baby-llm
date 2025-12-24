#----------------------------------------------------------------------------
#File:       safety_monitor.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Safety monitor for harm detection and ethical constraints
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Safety monitor for harm detection and ethical constraints.

Extracted from neural-child-2/safety_monitor.py
Adapted imports to use unified structure.
Extended with complete implementation including SafetyException and age-appropriateness checking.
"""

from typing import Dict, Any, Optional, List
import re
import logging

# Optional imports for unified structure
try:
    from neural_child.models.schemas import DevelopmentalStage
    DEVELOPMENTAL_STAGE_AVAILABLE = True
except ImportError:
    DEVELOPMENTAL_STAGE_AVAILABLE = False
    DevelopmentalStage = None
    print("Warning: DevelopmentalStage not available. Age-appropriateness checking will be limited.")

# Optional config
try:
    from neural_child.utils.config import EthicalConstraintsConfig
    ETHICAL_CONFIG_AVAILABLE = True
except ImportError:
    ETHICAL_CONFIG_AVAILABLE = False
    EthicalConstraintsConfig = None

# Configure logging
logger = logging.getLogger(__name__)


class SafetyException(Exception):
    """Exception raised when safety constraints are violated."""
    pass


class SafetyMonitor:
    """Safety monitor for harm detection and ethical constraints.
    
    Monitors interactions for:
    - Emotional stress levels
    - Age-appropriate content
    - Harmful language or concepts
    - Ethical constraint violations
    """
    
    def __init__(self, child_model: Optional[Any] = None, config: Optional[Any] = None):
        """Initialize the safety monitor.
        
        Args:
            child_model: Reference to the child model/brain (optional)
            config: Configuration object (optional)
        """
        self.child = child_model
        
        # Safety thresholds
        if ETHICAL_CONFIG_AVAILABLE and config and hasattr(config, 'ethical'):
            ethical_config = config.ethical
            self.safety_thresholds = {
                'stress_level': getattr(ethical_config, 'anxiety_threshold', 0.7),
                'emotional_stability': 0.3,
                'interaction_intensity': 0.8,
                'harm_threshold': getattr(ethical_config, 'harm_threshold', 0.4)
            }
        else:
            self.safety_thresholds = {
                'stress_level': 0.7,
                'emotional_stability': 0.3,
                'interaction_intensity': 0.8,
                'harm_threshold': 0.4
            }
        
        # Harmful words/phrases to detect
        self.harmful_patterns = [
            r'\b(violence|hurt|harm|danger|fear|scary|terrifying)\b',
            r'\b(kill|die|death|dead|murder)\b',
            r'\b(hate|destroy|break|damage)\b'
        ]
        
        # Age-appropriate complexity levels by stage
        self.complexity_limits = {
            'NEWBORN': {'max_words': 5, 'max_sentence_length': 10},
            'INFANT': {'max_words': 10, 'max_sentence_length': 15},
            'EARLY_TODDLER': {'max_words': 15, 'max_sentence_length': 20},
            'LATE_TODDLER': {'max_words': 20, 'max_sentence_length': 25},
            'EARLY_PRESCHOOL': {'max_words': 30, 'max_sentence_length': 35},
            'LATE_PRESCHOOL': {'max_words': 50, 'max_sentence_length': 50}
        }
        
    def monitor_interaction(self, message: str, child_state: Optional[Dict[str, Any]] = None) -> bool:
        """Monitor interaction safety.
        
        Args:
            message: The message to monitor
            child_state: Current child state dictionary (optional)
            
        Returns:
            True if interaction is safe
            
        Raises:
            SafetyException: If safety constraints are violated
        """
        # Check emotional state if provided
        if child_state:
            fear_level = child_state.get('fear', 0.0)
            if isinstance(fear_level, (int, float)) and fear_level > self.safety_thresholds['stress_level']:
                raise SafetyException(
                    f"Child is showing signs of stress (fear level: {fear_level:.2f}). "
                    "Please adjust interaction approach."
                )
        
        # Check message appropriateness
        if not self._is_age_appropriate(message):
            stage_name = self._get_current_stage_name()
            raise SafetyException(
                f"Message complexity not appropriate for "
                f"developmental stage: {stage_name}"
            )
        
        # Check for harmful content
        if self._contains_harmful_content(message):
            raise SafetyException(
                "Message contains potentially harmful content. "
                "Please use age-appropriate language."
            )
        
        return True
    
    def check_ethical_constraints(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action violates ethical constraints.
        
        Args:
            action: Action dictionary to check
            
        Returns:
            Dictionary with 'safe' (bool) and 'violations' (list) keys
        """
        violations = []
        
        # Check harm threshold
        harm_score = action.get('harm_score', 0.0)
        if harm_score > self.safety_thresholds['harm_threshold']:
            violations.append({
                'type': 'harm_threshold',
                'severity': 'high',
                'message': f'Harm score {harm_score:.2f} exceeds threshold {self.safety_thresholds["harm_threshold"]:.2f}'
            })
        
        # Check fairness
        fairness_score = action.get('fairness_score', 1.0)
        if fairness_score < 0.5:
            violations.append({
                'type': 'fairness',
                'severity': 'medium',
                'message': f'Fairness score {fairness_score:.2f} is below acceptable threshold'
            })
        
        return {
            'safe': len(violations) == 0,
            'violations': violations
        }
    
    def _is_age_appropriate(self, message: str) -> bool:
        """Check if message is age-appropriate.
        
        Args:
            message: Message to check
            
        Returns:
            True if message is age-appropriate
        """
        stage_name = self._get_current_stage_name()
        
        if stage_name not in self.complexity_limits:
            # Default to most restrictive if stage unknown
            stage_name = 'NEWBORN'
        
        limits = self.complexity_limits[stage_name]
        
        # Check word count
        words = message.split()
        if len(words) > limits['max_words']:
            logger.warning(f"Message has {len(words)} words, limit is {limits['max_words']} for {stage_name}")
            return False
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', message)
        for sentence in sentences:
            sentence_words = sentence.split()
            if len(sentence_words) > limits['max_sentence_length']:
                logger.warning(f"Sentence has {len(sentence_words)} words, limit is {limits['max_sentence_length']} for {stage_name}")
                return False
        
        return True
    
    def _contains_harmful_content(self, message: str) -> bool:
        """Check if message contains harmful content.
        
        Args:
            message: Message to check
            
        Returns:
            True if harmful content is detected
        """
        message_lower = message.lower()
        
        for pattern in self.harmful_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(f"Detected potentially harmful pattern: {pattern}")
                return True
        
        return False
    
    def _get_current_stage_name(self) -> str:
        """Get the current developmental stage name.
        
        Returns:
            Stage name as string
        """
        if self.child and hasattr(self.child, 'curriculum'):
            if hasattr(self.child.curriculum, 'current_stage'):
                stage = self.child.curriculum.current_stage
                if hasattr(stage, 'name'):
                    return stage.name
                elif isinstance(stage, str):
                    return stage
                elif DEVELOPMENTAL_STAGE_AVAILABLE and isinstance(stage, DevelopmentalStage):
                    return stage.name
        
        # Default to INFANT if unknown
        return 'INFANT'
    
    def update_safety_thresholds(self, thresholds: Dict[str, float]):
        """Update safety thresholds.
        
        Args:
            thresholds: Dictionary of threshold values to update
        """
        self.safety_thresholds.update(thresholds)
        logger.info(f"Updated safety thresholds: {thresholds}")
    
    def get_safety_status(self, child_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get current safety status.
        
        Args:
            child_state: Current child state (optional)
            
        Returns:
            Dictionary with safety status information
        """
        status = {
            'safe': True,
            'warnings': [],
            'thresholds': self.safety_thresholds.copy()
        }
        
        if child_state:
            fear_level = child_state.get('fear', 0.0)
            if fear_level > self.safety_thresholds['stress_level'] * 0.8:
                status['warnings'].append(f"High fear level: {fear_level:.2f}")
                status['safe'] = False
        
        return status


# Alias for backward compatibility
InteractionSafety = SafetyMonitor

