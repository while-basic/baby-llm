#----------------------------------------------------------------------------
#File:       stages.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Developmental stages for neural child development
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Developmental stages for neural child development.

Note: This module provides a local DevelopmentalStage enum for backward compatibility.
The unified DevelopmentalStage from neural_child.models.schemas should be used in production.

Extracted from neural-child-init/developmental_stages.py
Adapted to use unified structure.
"""

from enum import Enum, auto

# Import from unified schemas for consistency
from neural_child.models.schemas import DevelopmentalStage as UnifiedDevelopmentalStage

# Keep local enum for backward compatibility with existing code
class DevelopmentalStage(Enum):
    """Developmental stages for the neural child.

    Note: This is kept for backward compatibility. Use UnifiedDevelopmentalStage
    from neural_child.models.schemas in new code.
    """

    NEWBORN = auto()         # 0-3 months
    INFANT = auto()          # 3-6 months
    EARLY_TODDLER = auto()   # 6-12 months
    LATE_TODDLER = auto()    # 12-18 months
    EARLY_PRESCHOOL = auto() # 18-24 months
    LATE_PRESCHOOL = auto()  # 2-3 years
    EARLY_CHILDHOOD = auto() # 3-4 years
    MIDDLE_CHILDHOOD = auto() # 4-5 years
    LATE_CHILDHOOD = auto()  # 5-6 years
    PRE_ADOLESCENT = auto()  # 6-12 years
    EARLY_TEEN = auto()      # 12-14 years
    MID_TEEN = auto()        # 14-16 years
    LATE_TEEN = auto()       # 16-18 years
    YOUNG_ADULT = auto()     # 18-21 years
    EARLY_TWENTIES = auto()  # 21-25 years
    LATE_TWENTIES = auto()   # 25-30 years

# Export both for flexibility
__all__ = ['DevelopmentalStage', 'UnifiedDevelopmentalStage']

