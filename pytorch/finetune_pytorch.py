import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import logging
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Transformers imports
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.models.zamba2.modeling_zamba2 import (
    Zamba2ForCausalLM,
    Zamba2Config,
    HybridMambaAttentionDynamicCache
)

# Distributed Shampoo imports
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, GraftingType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DebugConfig:
    """Configuration for debugging and profiling"""
    enable_memory_profiling: bool = False
    enable_timing_profiling: bool = False
    enable_attention_debugging: bool = False
    print_model_summary: bool = False
    save_memory_traces: bool = False
    attention_impl: str = "auto"  # "auto", "pytorch", "flash_attn", "sdpa"
    profile_gpu_memory: bool = False
    debug_nan_inf: bool = True
    debug_gradients: bool = True

@dataclass
class Zamba2TrainingConfig:
    """Enhanced training configuration with debugging options"""
    # Base configs
    model_name: str = "Zyphra/Zamba2-2.7B"
    output_dir: str = "zamba2-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 1e-3
    max_seq_length: int = 512
    
    # Distributed Shampoo configs
    grafting_type: GraftingType = GraftingType.ADAM
    grafting_beta2: float = 0.999
    grafting_epsilon: float = 1e-8
    max_preconditioner_dim: int = 1024
    preconditioner_update_freq: int = 100
    start_preconditioning_step: int = 25
    
    # Training precision configs
    fp16: bool = True
    bf16: bool = False
    
    # Distributed training configs
    local_rank: int = field(default=-1, init=False)
    world_size: int = field(default=1, init=False)
    
    # Debug configs
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    def __post_init__(self):
        if torch.cuda.is_available():
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))

class MemoryProfiler:
    """Memory profiling utility"""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timestamps = []
        self.memory_stats = []

    def log_memory(self, tag: str):
        if not self.enabled:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            
            self.timestamps.append(time.time())
            self.memory_stats.append({
                'tag': tag,
                'allocated_mb': memory_allocated,
                'reserved_mb': memory_reserved
            })
            
            logger.info(f"Memory {tag}: Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")

class PureAttention(nn.Module):
    """Pure PyTorch implementation of attention for debugging"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.shape[0]
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size
        )
        
        return attn_output, attn_weights

class AttentionWrapper:
    """Wrapper to try different attention implementations"""
    @staticmethod
    def get_attention_module(config, debug_config):
        implementations = [
            ("flash_attn", lambda: FlashAttention2(config)),
            ("sdpa", lambda: SDPAttention(config)),
            ("pytorch", lambda: PureAttention(config))
        ]
        
        errors = {}
        for name, impl in implementations:
            if debug_config.attention_impl == "auto" or debug_config.attention_impl == name:
                try:
                    module = impl()
                    logger.info(f"Successfully initialized {name} attention")
                    return module
                except Exception as e:
                    errors[name] = str(e)
                    logger.warning(f"Failed to initialize {name} attention: {e}")
        
        raise RuntimeError(f"No attention implementation available. Errors: {errors}")

class Zamba2Dataset(Dataset):
    """Enhanced dataset with debugging options"""
    def __init__(self, texts: List[str], tokenizer, max_length: int, debug: bool = False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug
        self.tokenization_stats = [] if debug else None

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        tokenization_start = time.time()
        try:
            # Attempt 1: Standard tokenization
            encodings = self._try_standard_tokenization(text)
        except Exception as e:
            logger.warning(f"Standard tokenization failed: {e}")
            try:
                # Attempt 2: Chunked tokenization
                encodings = self._try_chunked_tokenization(text)
            except Exception as e:
                logger.warning(f"Chunked tokenization failed: {e}")
                # Attempt 3: Basic tokenization
                encodings = self._try_basic_tokenization(text)

        if self.debug:
            self.tokenization_stats.append({
                'idx': idx,
                'text_length': len(text),
                'tokens_length': len(encodings['input_ids']),
                'time': time.time() - tokenization_start
            })

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze().clone()
        }

    def _try_standard_tokenization(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def _try_chunked_tokenization(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True
        )

    def _try_basic_tokenization(self, text):
        encodings = {
            "input_ids": self.tokenizer.encode(
                text, 
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            ),
            "attention_mask": [1] * min(len(text), self.max_length)
        }
        return {k: torch.tensor(v) for k, v in encodings.items()}

    def __len__(self):
        return len(self.texts)