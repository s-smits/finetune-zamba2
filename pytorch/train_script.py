from typing import List, Optional, Dict

import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from .zamba2_dataset import Zamba2Dataset
from .zamba2_model import Zamba2Model
from .zamba2_config import Zamba2Config
from .zamba2_training_config import Zamba2TrainingConfig ## MISSING, STILL NEEDS TO BE ADDED
from .zamba2_for_causal_lm import Zamba2ForCausalLM ## MISSING, STILL NEEDS TO BE ADDED
from .zamba2_attention_decoder_layer import Zamba2AttentionDecoderLayer ## MISSING, STILL NEEDS TO BE ADDED
from .zamba2_mamba_decoder_layer import Zamba2MambaDecoderLayer

import wandb
from attention_factory import AttentionFactory, AttentionWithFallback

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_attention_layers(
    model: Zamba2ForCausalLM,
    config: Zamba2TrainingConfig
) -> Zamba2ForCausalLM:
    """Initialize attention layers with fallback options"""
    try:
        # Get number of layers that need attention
        num_attention_layers = sum(1 for block_type in model.config.layers_block_type if block_type == "g")
        
        attention_layers = []
        for layer_idx in range(num_attention_layers):
            try:
                if config.enable_attention_fallback:
                    attention = AttentionWithFallback(
                        config=model.config,
                        layer_idx=layer_idx,
                        num_mem_blocks=num_attention_layers
                    )
                    logger.info(f"Initialized attention layer {layer_idx} with runtime fallbacks")
                else:
                    if config.attention_type:
                        attention = AttentionFactory.create_specific_attention(
                            attention_type=config.attention_type,
                            config=model.config,
                            layer_idx=layer_idx,
                            num_mem_blocks=num_attention_layers
                        )
                    else:
                        attention, impl_type = AttentionFactory.create_attention(
                            config=model.config,
                            layer_idx=layer_idx,
                            num_mem_blocks=num_attention_layers
                        )
                    logger.info(f"Initialized attention layer {layer_idx} with {config.attention_type or impl_type}")
                
                attention_layers.append(attention)
            except Exception as e:
                logger.error(f"Failed to initialize attention layer {layer_idx}: {str(e)}")
                raise

        # Replace attention layers in model
        for i, layer in enumerate(model.model.blocks):
            if hasattr(layer, 'self_attn'):
                layer.self_attn = attention_layers[i]
        
        return model
    except Exception as e:
        logger.error(f"Failed to initialize attention layers: {str(e)}")
        raise

class Zamba2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
        self.attention_fallback_counts: Dict[str, int] = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with attention fallback tracking"""
        try:
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Track successful attention implementations
            impl_types = model.get_current_attention_impl_types()
            for impl_type in impl_types:
                self.attention_fallback_counts[impl_type] = \
                    self.attention_fallback_counts.get(impl_type, 0) + 1
            
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logger.warning(f"Loss computation failed: {str(e)}")
            raise

    def log_metrics(self, split, metrics):
        """Add attention implementation metrics"""
        super().log_metrics(split, metrics)
        
        if self.attention_fallback_counts:
            attention_metrics = {
                f"attention_{impl}_count": count
                for impl, count in self.attention_fallback_counts.items()
            }
            self.log(attention_metrics)

def count_mem_blocks_in_config(config: Zamba2Config) -> int:
    """Count number of attention blocks needed"""
    return sum(1 for block_type in config.layers_block_type if block_type == "g")

def layer_type_list(config: Zamba2Config) -> List[int]:
    """Get indices of attention layers"""
    return [i for i, block_type in enumerate(config.layers_block_type) if block_type == "g"]

def setup_distributed():
    """Setup distributed training if multiple GPUs are available"""
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("nccl")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            return True, world_size, local_rank
    return False, 1, 0

def train_zamba2_model(
    config: Zamba2TrainingConfig,
    train_texts: List[str],
    eval_texts: Optional[List[str]] = None,
):
    # Set random seed
    set_seed(42)

    # Setup distributed training
    is_distributed, world_size, local_rank = setup_distributed()
    config.world_size = world_size
    config.local_rank = local_rank

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and move to device
    model = Zamba2ForCausalLM(config)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Initialize wandb if main process
    if local_rank in [-1, 0]:
        try:
            wandb.init(project="zamba2-training", config=vars(config))
        except Exception as e:
            logger.warning(f"WandB initialization failed: {str(e)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with custom config
    model_config = Zamba2Config.from_pretrained(config.model_name)
    model_config.use_flash_attention = config.use_flash_attention
    model_config.use_mem_rope = config.use_mem_rope
    model_config.rope_theta = config.rope_theta
    model_config.sliding_window = config.sliding_window
    model_config.attention_dropout = config.attention_dropout

    try:
        # Initialize base model first
        base_model = Zamba2Model(model_config)
        
        # Initialize attention layers
        num_gs = count_mem_blocks_in_config(model_config)
        
        # Initialize transformer blocks with attention
        blocks = []
        for layer_idx in range(num_gs):
            try:
                if config.enable_attention_fallback:
                    attention = AttentionWithFallback(
                        config=model_config,
                        layer_idx=layer_idx,
                        num_mem_blocks=num_gs
                    )
                else:
                    attention, impl_type = AttentionFactory.create_attention(
                        config=model_config,
                        layer_idx=layer_idx,
                        num_mem_blocks=num_gs
                    )
                
                # Create decoder layer with attention
                decoder_layer = Zamba2AttentionDecoderLayer(model_config)
                decoder_layer.self_attn = attention
                blocks.append(decoder_layer)
                
            except Exception as e:
                logger.error(f"Failed to initialize attention block {layer_idx}: {str(e)}")
                raise

        # Initialize Mamba layers
        mamba_layers = []
        linear_layers = []
        for i in range(model_config.num_hidden_layers):
            if model_config.layers_block_type[i] == "m":
                mamba_layers.append(Zamba2MambaDecoderLayer(model_config, layer_idx=i))
            elif model_config.layers_block_type[i] == "g":
                linear_layers.append(nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=False))
                mamba_layers.append(Zamba2MambaDecoderLayer(model_config, layer_idx=i))

        # Set model components
        base_model.blocks = nn.ModuleList(blocks)
        base_model.mamba_layers = nn.ModuleList(mamba_layers)
        base_model.linear_layers = nn.ModuleList(linear_layers)
        
        # Create CausalLM model
        model = Zamba2ForCausalLM(model_config)
        model.model = base_model
        
        # Initialize weights
        model.post_init()
        
        # Move model to correct device and dtype
        model = model.to(
            dtype=torch.bfloat16 if getattr(config, 'bf16', False) else torch.float16,
            device=device
        )
        
        logger.info("Successfully initialized model with attention and mamba layers")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

    # Prepare datasets
    train_dataset = Zamba2Dataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    eval_dataset = None
    if eval_texts:
        eval_dataset = Zamba2Dataset(
            texts=eval_texts,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
        )

    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        evaluation_strategy=config.evaluation_strategy if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        fp16=getattr(config, 'fp16', False),
        bf16=getattr(config, 'bf16', False),
        local_rank=config.local_rank,
        gradient_checkpointing=True,
        group_by_length=True,
        remove_unused_columns=False,
    )

    # Initialize trainer with custom components
    trainer = Zamba2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    if local_rank in [-1, 0]:
        # Save attention implementation info
        impl_types = model.get_current_attention_impl_types()
        model.config.attention_implementation = impl_types
        
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        wandb.finish()

    # Cleanup
    if torch.cuda.is_available() and config.world_size > 1:
        dist.destroy_process_group()

def main():
    config = Zamba2TrainingConfig(
        model_name="Zyphra/Zamba2-2.7B",
        output_dir="zamba2-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        max_seq_length=512,
    )
    
    train_texts = [
        "Your training text 1",
        "Your training text 2"
    ]
    
    eval_texts = [
        "Your evaluation text 1",
        "Your evaluation text 2"
    ]
    
    try:
        train_zamba2_model(
            config=config,
            train_texts=train_texts,
            eval_texts=eval_texts,
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Use torchrun for distributed training
    main()