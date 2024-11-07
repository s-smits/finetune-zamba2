import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Tuple, Dict, Any
from transformers.models.zamba2.modeling_zamba2 import Zamba2Config
from attn_impl.zamba2_attention import Zamba2Attention
from attn_impl.zamba2_flash_attention2 import Zamba2FlashAttention2
from attn_impl.zamba2_sdpa_attention import Zamba2SdpaAttention

logger = logging.getLogger(__name__)

class AttentionFactory:
    """Factory class to manage different attention implementations with fallbacks"""
    
    ATTENTION_IMPLS = {
        "flash_attention_2": (Zamba2FlashAttention2, "Flash Attention 2"),
        "sdpa": (Zamba2SdpaAttention, "SDPA Attention"),
        "eager": (Zamba2Attention, "Base Attention")
    }

    @staticmethod
    def create_attention(
        config: Zamba2Config,
        layer_idx: Optional[int] = None,
        num_mem_blocks: Optional[int] = None
    ) -> Tuple[nn.Module, str]:
        """Try different attention implementations in order"""
        errors = []

        for impl_name, (impl_class, impl_desc) in AttentionFactory.ATTENTION_IMPLS.items():
            try:
                attention = impl_class(
                    config=config,
                    layer_idx=layer_idx,
                    num_mem_blocks=num_mem_blocks
                )
                logger.info(f"Successfully initialized {impl_desc}")
                return attention, impl_name
            except (RuntimeError, ImportError, ValueError) as e:
                error_msg = f"{impl_desc} initialization failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        raise RuntimeError(f"All attention implementations failed:\n" + "\n".join(errors))

    @staticmethod
    def create_specific_attention(
        attention_type: str,
        config: Zamba2Config,
        layer_idx: Optional[int] = None,
        num_mem_blocks: Optional[int] = None
    ) -> nn.Module:
        """Create specific attention implementation with fallbacks"""
        if attention_type not in AttentionFactory.ATTENTION_IMPLS:
            raise ValueError(f"Unknown attention type: {attention_type}")

        impl_class, impl_desc = AttentionFactory.ATTENTION_IMPLS[attention_type]
        try:
            attention = impl_class(
                config=config,
                layer_idx=layer_idx,
                num_mem_blocks=num_mem_blocks
            )
            logger.info(f"Successfully initialized {impl_desc}")
            return attention
        except Exception as e:
            error_msg = f"{impl_desc} initialization failed: {str(e)}"
            logger.warning(error_msg)
            
            # Try fallbacks
            for fallback_type, (fallback_class, fallback_desc) in AttentionFactory.ATTENTION_IMPLS.items():
                if fallback_type == attention_type:
                    continue
                try:
                    attention = fallback_class(
                        config=config,
                        layer_idx=layer_idx,
                        num_mem_blocks=num_mem_blocks
                    )
                    logger.info(f"Successfully fell back to {fallback_desc}")
                    return attention
                except Exception as fallback_e:
                    logger.warning(f"Fallback to {fallback_desc} failed: {str(fallback_e)}")

            raise RuntimeError("All attention implementations failed")


class AttentionWithFallback(nn.Module):
    """Wrapper class for attention with runtime fallbacks"""
    
    def __init__(
        self,
        config: Zamba2Config,
        layer_idx: Optional[int] = None,
        num_mem_blocks: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_mem_blocks = num_mem_blocks
        
        # Initialize attention implementations
        self.attention_impls: Dict[str, nn.Module] = {}
        self.initialize_attention_impls()
        
        if not self.attention_impls:
            raise RuntimeError("No attention implementations available")

        # Set default implementation
        self.current_impl_type, self.current_impl = next(iter(self.attention_impls.items()))

    def initialize_attention_impls(self):
        """Initialize all available attention implementations"""
        for impl_type, (impl_class, impl_desc) in AttentionFactory.ATTENTION_IMPLS.items():
            try:
                attention = impl_class(
                    config=self.config,
                    layer_idx=self.layer_idx,
                    num_mem_blocks=self.num_mem_blocks
                )
                self.attention_impls[impl_type] = attention
                logger.info(f"Initialized {impl_desc}")
            except Exception as e:
                logger.warning(f"Failed to initialize {impl_desc}: {str(e)}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """Forward pass with fallback mechanism"""
        try:
            return self.current_impl(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        except Exception as e:
            logger.warning(f"Forward pass failed with {self.current_impl_type}: {str(e)}")
            
            # Try other implementations
            for impl_type, impl in self.attention_impls.items():
                if impl_type == self.current_impl_type:
                    continue
                try:
                    output = impl(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    logger.info(f"Successfully fell back to {impl_type}")
                    self.current_impl = impl
                    self.current_impl_type = impl_type
                    return output
                except Exception as fallback_e:
                    logger.warning(f"Fallback to {impl_type} failed: {str(fallback_e)}")

            raise RuntimeError("All attention implementations failed in forward pass")