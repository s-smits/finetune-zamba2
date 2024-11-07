import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Tuple, Any
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.zamba2.configuration_zamba2 import Zamba2Config
from transformers.models.zamba2.modeling_zamba2 import Zamba2PreTrainedModel
from .zamba2_attention_decoder_layer import Zamba2AttentionDecoderLayer
from .zamba2_mamba_decoder_layer import Zamba2MambaDecoderLayer
from .zamba2_rmsnorm import Zamba2RMSNorm
from .hybrid_mamba_attention_dynamic_cache import HybridMambaAttentionDynamicCache

logger = logging.getLogger(__name__)

@add_start_docstrings(
    "The bare Zamba2 Model outputting raw hidden-states without any specific head on top.",
    ZAMBA2_START_DOCSTRING,
)
class Zamba2Model(Zamba2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Zamba2DecoderLayer`]

    Args:
        config: Zamba2Config
    """
    def __init__(self, config: Zamba2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        if config.use_long_context:
            logger.warning_once(
                    f"`use_long_context` has been set to True, therefore `max_position_embeddings` will be set to 16384."
                )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.blocks = torch.nn.ModuleList([Zamba2AttentionDecoderLayer(config) for _ in range(config.num_mem_blocks)])
        mamba_layers = []
        linear_layers = []
        self.layers_block_type = config.layers_block_type
        for i in range(config.num_hidden_layers):
            if config.layers_block_type[i] == "m":
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
            elif config.layers_block_type[i] == "g":
                linear_layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False))
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.linear_layers = nn.ModuleList(linear_layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(ZAMBA2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        
        original_hidden_states = torch.clone(inputs_embeds)
        # original_hidden_states: word embedding output that will be concatenated with hidden activations to form the input of the shared transformer layer
        if use_cache and past_key_values is None:
            logger.warning_once(
                "Zamba2 requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        mamba_layers = iter(self.mamba_layers)
        linear_layers = iter(self.linear_layers)
        block_count = 0
        for layer_idx, layer_type in enumerate(self.layers_block_type):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if layer_type == "g":
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        self.blocks[block_count % self.config.num_mem_blocks].__call__,
                        hidden_states,
                        original_hidden_states,
                        block_count,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        block_count,
                    )
                else:
                    layer_outputs = self.blocks[block_count % self.config.num_mem_blocks](
                        hidden_states,
                        original_hidden_states=original_hidden_states,
                        layer_idx=block_count,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                block_count += 1
                transformer_hidden_states = layer_outputs[0]
                if output_attentions:
                    if layer_outputs[1] is not None:
                        all_self_attns += (layer_outputs[1],)
                if self.gradient_checkpointing and self.training:
                    transformer_hidden_states = self._gradient_checkpointing_func(
                        next(linear_layers).__call__,
                        transformer_hidden_states,
                    )
                else:
                    transformer_hidden_states = next(linear_layers)(
                        transformer_hidden_states,
                    )
            else:
                transformer_hidden_states = None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    next(mamba_layers).__call__,
                    hidden_states,
                    transformer_hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = next(mamba_layers)(
                    hidden_states,
                    transformer_hidden_states=transformer_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]
            
        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    # Copied from transformers.models.jamba.modeling_jamba.JambaModel._update_causal_mask
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
