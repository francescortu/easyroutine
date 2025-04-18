import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math
from transformers import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from easyroutine.interpretability.module_wrappers.base import (
    BaseAttentionWrapper,
    AttentionMatrixHookModule,
)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim=1,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (batch, num_key_value_heads, seq_len, head_dim)
        -> (batch, num_attention_heads, seq_len, head_dim)
    """
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)

class Gemma3AttentionWrapper(BaseAttentionWrapper):
    """
    A wrapper for the original Gemma3Attention that adds an attention hook.
    It replicates all relevant submodules/attributes of the original module so
    that its forward pass follows the same strategy as your Llama attention.
    """
    @staticmethod
    def original_name():
        return "Gemma3Attention"

    def __init__(self, original_attention: nn.Module):
        """
        Copies references to the original Gemma3Attention submodules/attributes.
        """
        super().__init__(original_attention)
        # Keep a private reference to the original module.
        object.__setattr__(self, "_orig_attn", original_attention)

        # Replicate the original attention's submodules.
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        self.q_norm = original_attention.q_norm
        self.k_norm = original_attention.k_norm

        # Copy scalar attributes.
        self.head_dim = original_attention.head_dim
        self.scaling = original_attention.scaling
        self.attention_dropout = original_attention.attention_dropout
        self.layer_idx = original_attention.layer_idx
        self.config = original_attention.config
        self.sliding_window = original_attention.sliding_window
        self.num_key_value_groups = original_attention.num_key_value_groups

        # Add the custom hook module.
        self.attention_matrix_pre_softmax_hook = AttentionMatrixHookModule()
        self.attention_matrix_hook = AttentionMatrixHookModule()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Prepare shapes.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute query, key, value states.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RMSNorm.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary positional embeddings.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle caching if provided.
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states = key_states[:, :, :seq_len, :]
                value_states = value_states[:, :, :seq_len, :]

        # Inline the eager attention forward logic.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute scaled dot-product attention logits.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply softmax.
        attn_weights = self.attention_matrix_pre_softmax_hook(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Pass the attention weights through the hook.
        attn_weights = self.attention_matrix_hook(attn_weights)
        attn_weights = F.dropout(
            attn_weights,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )

        # Compute attention output.
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights  # type: ignore
