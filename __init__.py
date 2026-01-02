"""
Conversational LLM Trainer
==========================
FP8 training framework for RTX 5090 Blackwell with CUDA graphs.
"""

from .model import SmallLM, ModelConfig
from .dataset import DatasetLoader, DatasetConfig, ConversationTokenizer, TrainingDataset

__version__ = "1.0.0"
__all__ = [
    "SmallLM",
    "ModelConfig", 
    "DatasetLoader",
    "DatasetConfig",
    "ConversationTokenizer",
    "TrainingDataset",
]
