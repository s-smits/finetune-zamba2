"""
finetune-zamba2: A package for fine-tuning the Zamba2 language model
"""
# Import main components to make them available at package level
from .finetune import CustomTrainer, main
from .finetune_config import CONFIG, HF_TOKEN
from .lr_optimizer import train_with_lr_optimization

from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"

# Define package base directory
PACKAGE_ROOT = Path(__file__).parent.resolve()


# Define what should be available when using `from finetune-zamba2 import *`
__all__ = [
    'CustomTrainer',
    'main',
    'CONFIG',
    'HF_TOKEN',
    'train_with_lr_optimization',
    'PACKAGE_ROOT'
]