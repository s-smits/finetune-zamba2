class Zamba2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Zamba2Config, layer_idx: Optional[int] = None, num_mem_blocks = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_mem_blocks = num_mem_blocks
        self.rope_theta = config.rope_theta

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = 2 * self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.layer_block_map = layer_type_list(config)
        

        if (self.head_dim * self.num_heads) != 2 * self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(2 * self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if config.use_shared_attention_lora:
            self.linear_q_lora_A_list = nn.ParameterList([])
            self.linear_q_lora_B_list = nn.ParameterList([])
            self.linear_k_lora_A_list = nn.ParameterList([])
            self.linear_k_lora_B_list = nn.ParameterList([])
            self.linear_v_lora_A_list = nn.ParameterList([])
            self.linear_v_lora_B_list = nn.ParameterList([])
            
            for i in range(self.num_mem_blocks):
                # we store all loras in a list
                linear_q_lora_A = nn.Linear(2 * self.config.hidden_size, self.config.lora_rank, bias = False)
                linear_q_lora_B = nn.Linear(self.config.lora_rank, 2 * self.config.hidden_size, bias = False)
                self.linear_q_lora_A_list.append(linear_q_lora_A)
                self.linear_q_lora_B_list.append(linear_q_lora_B)
                linear_k_lora_A = nn.Linear(2 * self.config.hidden_size, self.config.lora_rank, bias = False)
                linear_k_lora_B = nn.Linear(self.config.lora_rank, 2 * self.config.hidden_size, bias = False)
                self.linear_k_lora_A_list.append(linear_k_lora_A)
                self.linear_k_lora_B_list.append(linear_k_lora_B)
                linear_v_lora_A = nn.Linear(2 * self.config.hidden_size, self.config.lora_rank, bias = False)
                linear_v_lora_B = nn.Linear(self.config.lora_rank, 2 * self.config.hidden_size, bias = False)
                self.linear_v_lora_A_list.append(linear_v_lora_A)
                self.linear_v_lora_B_list.append(linear_v_lora_B)

        if config.use_mem_rope:
            self.rotary_emb = Zamba2RotaryEmbedding(
                config,
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=self.rope_theta,
            )



    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.use_shared_attention_lora:
            linear_q_lora_A = self.linear_q_lora_A_list[layer_idx]
            linear_q_lora_B = self.linear_q_lora_B_list[layer_idx]
            q_lora_output = linear_q_lora_A(hidden_states)
            q_lora_output = linear_q_lora_B(q_lora_output)
            query_states = self.q_proj(hidden_states)
            query_states = query_states + q_lora_output
            linear_k_lora_A = self.linear_k_lora_A_list[layer_idx]
            linear_k_lora_B = self.linear_k_lora_B_list[layer_idx]
            k_lora_output = linear_k_lora_A(hidden_states)
            k_lora_output = linear_k_lora_B(k_lora_output)
            key_states = self.k_proj(hidden_states)
            key_states = key_states + k_lora_output
            linear_v_lora_A = self.linear_v_lora_A_list[layer_idx]
            linear_v_lora_B = self.linear_v_lora_B_list[layer_idx]
            v_lora_output = linear_v_lora_A(hidden_states)
            v_lora_output = linear_v_lora_B(v_lora_output)
            value_states = self.v_proj(hidden_states)
            value_states = value_states + v_lora_output
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            layer_idx = self.layer_block_map[layer_idx]
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim / 2)
     
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, 2 * self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = attn_output

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
