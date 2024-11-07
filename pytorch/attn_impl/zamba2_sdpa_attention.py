class Zamba2SdpaAttention(Zamba2Attention):
    """
    Zamba2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Zamba2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Zamba2Model is using Zamba2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

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

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        softmax_scale = 1 / (query_states.shape[-1] / 2) ** 0.5

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
            scale=softmax_scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, 2 * self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


ZAMBA2_ATTENTION_CLASSES = {
    "eager": Zamba2Attention,
    "flash_attention_2": Zamba2FlashAttention2,
    "sdpa": Zamba2SdpaAttention,
