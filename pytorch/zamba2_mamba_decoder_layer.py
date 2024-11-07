import torch
import torch.nn as nn
from typing import Optional, Tuple
from .zamba2_config import Zamba2Config
from .zamba2_rmsnorm import Zamba2RMSNorm
from .hybrid_attention_dynamic_cache import HybridMambaAttentionDynamicCache
from .mamba2_layer import Mamba2Layer

class Zamba2MambaDecoderLayer(nn.Module):
    def __init__(self, config: Zamba2Config, layer_idx: int):
        super().__init__()
        factory_kwargs = {}
        self.mamba = Mamba2Layer(config=config, layer_idx=layer_idx, **factory_kwargs)
        self.input_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        transformer_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        residual = hidden_states

        # `transformer_hidden_states` is the output from shared transformer + linear layer (see fig. 2 in https://arxiv.org/pdf/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba layer below (as described in eq. (6) of https://arxiv.org/pdf/2405.16712).
        hidden_states = (
            hidden_states + transformer_hidden_states if transformer_hidden_states is not None else hidden_states
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            u=hidden_states,
            inference_params=past_key_value,
            attention_mask=attention_mask,
        )

        self_attn_weights = None

        # residual connection after mamba
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs