import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.zamba2.modeling_zamba2 import Zamba2Config
from attention_factory import Zamba2Attention

class Zamba2MLP(nn.Module):

    def __init__(self, config: Zamba2Config,is_expert: bool = False, layer_idx=None, num_mem_blocks = None):
        super().__init__()

        self.num_mem_blocks = num_mem_blocks
        
        self.config: Zamba2Config = config
        self.layer = layer_idx
        ffn_hidden_size_1 = self.config.ffn_hidden_size
        ffn_hidden_size_2 = self.config.ffn_hidden_size
        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size_1 *= 2

        if self.layer == -1:
            ffn_hidden_size_1 = 8 * self.config.hidden_size

        self.linear_fc1 = nn.Linear(self.config.hidden_size, ffn_hidden_size_1, bias = self.config.add_bias_linear)
        if self.config.gated_linear_unit or self.layer == -1:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)

                return F.gelu(x[0]) * x[1]
            self.activation_func = glu
        else:
            self.activation_func = F.gelu


        self.linear_fc2 = nn.Linear(ffn_hidden_size_2, self.config.hidden_size, bias = self.config.add_bias_linear)
        
        if self.config.use_shared_block_lora:
            self.linear_fc1_lora_A_list = nn.ParameterList([])
            self.linear_fc1_lora_B_list = nn.ParameterList([])
            for i in range(self.num_mem_blocks):
                linear_fc1_lora_A = nn.Linear(self.config.hidden_size, self.config.lora_rank, bias = False)
                linear_fc1_lora_B = nn.Linear(self.config.lora_rank, ffn_hidden_size_1, bias = False)
                self.linear_fc1_lora_A_list.append(linear_fc1_lora_A)
                self.linear_fc1_lora_B_list.append(linear_fc1_lora_B)

    def forward(self, hidden_states, inference_params=None, forward_layer_idx = None):

        # [s, b, 4 * h/p]
        if self.config.use_shared_block_lora:
            linear_fc1_lora_A = self.linear_fc1_lora_A_list[forward_layer_idx]
            linear_fc1_lora_B = self.linear_fc1_lora_B_list[forward_layer_idx]
            lora_output = linear_fc1_lora_A(hidden_states)
            lora_output= linear_fc1_lora_B(lora_output)
            intermediate_parallel = self.linear_fc1(hidden_states)
            intermediate_parallel = intermediate_parallel + lora_output
        else:
            intermediate_parallel= self.linear_fc1(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.linear_fc2(intermediate_parallel)

        return output


class Zamba2AttentionDecoderLayer(nn.Module):
    def __init__(self, config: Zamba2Config, layer_idx: Optional[int] = None):
        super().__init__()
        num_gs = count_mem_blocks_in_config(config)
        self.self_attn = ZAMBA2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=-1, num_mem_blocks = num_gs)
        self.feed_forward = Zamba2MLP(config, layer_idx=-1, num_mem_blocks = num_gs)
        self.input_layernorm = Zamba2RMSNorm(2 * config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # The argument original_hidden_states is concatenated with hidden_states (which is the output of the previous (mamba) layer)
    # The concatenated tensor is then used as input of the pre-attention RMSNorm (see fig. 2 in https://arxiv.org/pdf/2405.16712).
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`
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
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, position_ids=position_ids, layer_idx=layer_idx)

        # feed-forward (MLP)
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, forward_layer_idx=layer_idx)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


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


ZAMBA2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Zamba2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
