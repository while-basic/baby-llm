"""NeuralChild: A psychological brain simulation framework.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

__version__ = "1.0.0"
__author__ = "Celaya Solutions AI Research Lab"
__license__ = "MIT"

from neuralchild.config import Config, load_config, get_config
from neuralchild.mind.mind_core import Mind
from neuralchild.mother.mother_llm import MotherLLM
from neuralchild.core.schemas import (
    DevelopmentalStage,
    NetworkMessage,
    Memory,
    Belief,
    Need,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Config",
    "load_config",
    "get_config",
    "Mind",
    "MotherLLM",
    "DevelopmentalStage",
    "NetworkMessage",
    "Memory",
    "Belief",
    "Need",
]
