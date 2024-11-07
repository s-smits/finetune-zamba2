"""
Zamba2 PyTorch Implementation
A hybrid architecture combining Mamba and Transformer blocks
"""

# Version
__version__ = "0.1.0"

# Core model components
from .zamba2_model import Zamba2Model
from .zamba2_config import Zamba2Config
from .zamba2_mlp import Zamba2MLP
from .zamba2_attention import Zamba2Attention
from .zamba2_rms_norm import Zamba2RMSNorm, Zamba2RotaryEmbedding
from .zamba2_mamba_decoder_layer import Zamba2MambaDecoderLayer
from .hybrid_attention_dynamic_cache import HybridMambaAttentionDynamicCache

# Training components
from .zamba2_dataset import Zamba2Dataset
from .attention_factory import (
    AttentionFactory,
    AttentionWithFallback,
)

# Training utilities
from .train_script import train_zamba2_model
from .finetune_pytorch import (
    DebugConfig,
    Zamba2TrainingConfig,
    MemoryProfiler,
    PureAttention,
    AttentionWrapper,
)

# Inference utilities
from .inference import generate_text

__all__ = [
    # Core model
    "Zamba2Model",
    "Zamba2Config",
    "Zamba2MLP",
    "Zamba2Attention",
    "Zamba2RMSNorm",
    "Zamba2RotaryEmbedding",
    "Zamba2MambaDecoderLayer",
    "HybridMambaAttentionDynamicCache",
    
    # Training
    "Zamba2Dataset",
    "AttentionFactory",
    "AttentionWithFallback",
    "train_zamba2_model",
    
    # Training configs and utilities
    "DebugConfig",
    "Zamba2TrainingConfig",
    "MemoryProfiler",
    "PureAttention",
    "AttentionWrapper",
    
    # Inference
    "generate_text",
]

# Optional: Set up logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())