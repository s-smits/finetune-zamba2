from dataclasses import dataclass
from typing import Optional, List
from transformers import PretrainedConfig

@dataclass
class Zamba2TrainingConfig(PretrainedConfig):
    """
    Configuration class for Zamba2 training parameters.

    Args:
        model_name (`str`):
            Name or path of the pretrained model to use.
        output_dir (`str`):
            Directory where model checkpoints and results will be saved.
        num_train_epochs (`int`):
            Total number of training epochs.
        per_device_train_batch_size (`int`):
            Batch size per device during training.
        per_device_eval_batch_size (`int`, *optional*):
            Batch size per device during evaluation.
        gradient_accumulation_steps (`int`):
            Number of steps to accumulate gradients before performing a backward pass.
        learning_rate (`float`):
            Initial learning rate.
        max_seq_length (`int`):
            Maximum sequence length for training.
        world_size (`int`, *optional*, defaults to 1):
            Number of processes for distributed training.
        local_rank (`int`, *optional*, defaults to -1):
            Local rank for distributed training.
        attention_type (`str`, *optional*):
            Type of attention to use ("flash_attention_2", "sdpa", "eager", or None for auto).
        enable_attention_fallback (`bool`, *optional*, defaults to True):
            Whether to enable runtime fallbacks for attention mechanisms.
        use_flash_attention (`bool`, *optional*, defaults to True):
            Whether to use Flash Attention when available.
        use_mem_rope (`bool`, *optional*, defaults to True):
            Whether to use memory-efficient RoPE implementation.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period for rotary position embeddings.
        sliding_window (`int`, *optional*):
            Size of the sliding window for attention, if used.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        warmup_steps (`int`, *optional*, defaults to 1000):
            Number of steps for learning rate warmup.
        weight_decay (`float`, *optional*, defaults to 0.01):
            Weight decay coefficient.
        fp16 (`bool`, *optional*, defaults to False):
            Whether to use 16-bit floating point precision.
        bf16 (`bool`, *optional*, defaults to False):
            Whether to use brain floating point precision.
        evaluation_strategy (`str`, *optional*, defaults to "steps"):
            Evaluation strategy to adopt during training.
        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations.
        save_steps (`int`, *optional*, defaults to 1000):
            Number of updates steps before saving a model checkpoint.
        logging_dir (`str`, *optional*, defaults to "./logs"):
            Directory for storing logs.
        logging_steps (`int`, *optional*, defaults to 100):
            Number of update steps between two logs.
    """

    model_type = "zamba2_training"

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        max_seq_length: int,
        per_device_eval_batch_size: Optional[int] = None,
        world_size: int = 1,
        local_rank: int = -1,
        attention_type: Optional[str] = None,
        enable_attention_fallback: bool = True,
        use_flash_attention: bool = True,
        use_mem_rope: bool = True,
        rope_theta: float = 10000.0,
        sliding_window: Optional[int] = None,
        attention_dropout: float = 0.0,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        fp16: bool = False,
        bf16: bool = False,
        evaluation_strategy: str = "steps",
        eval_steps: Optional[int] = None,
        save_steps: int = 1000,
        logging_dir: str = "./logs",
        logging_steps: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size or per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.world_size = world_size
        self.local_rank = local_rank

        # Attention-specific configs
        self.attention_type = attention_type
        self.enable_attention_fallback = enable_attention_fallback
        self.use_flash_attention = use_flash_attention
        self.use_mem_rope = use_mem_rope
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

        # Training-specific configs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.bf16 = bf16
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps

    @property
    def training_args_dict(self):
        """
        Returns a dictionary of training arguments compatible with HuggingFace's TrainingArguments.
        """
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "local_rank": self.local_rank,
            "gradient_checkpointing": True,
            "group_by_length": True,
            "remove_unused_columns": False,
        }

    def to_dict(self):
        """
        Converts the config object to a dictionary.
        """
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "attention_type": self.attention_type,
            "enable_attention_fallback": self.enable_attention_fallback,
            "use_flash_attention": self.use_flash_attention,
            "use_mem_rope": self.use_mem_rope,
            "rope_theta": self.rope_theta,
            "sliding_window": self.sliding_window,
            "attention_dropout": self.attention_dropout,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Creates a config object from a dictionary.
        """
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str):
        """
        Save a configuration object to the directory `save_directory`.
        """
        if self.local_rank in [-1, 0]:  # Save only on main process
            super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a configuration object from a pretrained model configuration.
        """
        config_dict = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(**config_dict) 