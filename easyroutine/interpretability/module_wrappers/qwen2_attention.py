import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable
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

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    attention_hook: Optional[nn.Module] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if attention_hook is not None:
        attn_weights = attention_hook(attn_weights)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class Qwen2AttentionWrapper(BaseAttentionWrapper):
    """
    A wrapper around the original LlamaAttention. It has:
    - The same named attributes (q_proj, k_proj, etc.), which are references
        to the original module's submodules/parameters.
    - A private reference (`_orig_attn`) to the entire original attention,
        for falling back if something isn't found on the wrapper itself.
    - An additional `attention_matrix_hook` for intercepting attention.
    """

    @staticmethod
    def original_name():
        return "Qwen2Attention"

    def __init__(self, original_attention: nn.Module):
        """
        Store references to all relevant submodules so the wrapper
        "feels" the same. Also store a reference to the original module
        in a private attribute for fallback.
        """
        super().__init__(original_attention)

        # This is the private reference to the entire original attention.
        # We'll fallback to it for any attribute we haven't explicitly set.
        object.__setattr__(self, "_orig_attn", original_attention)

        # Now replicate the original attention's submodules as attributes of *this* wrapper.
        # These are direct references, not new modules:
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj

        # Copy over any scalar attributes you need
        # self.num_heads = original_attention.num_heads
        # self.num_key_value_heads = original_attention.num_key_value_heads
        # self.num_key_value_groups = original_attention.num_key_value_groups
        self.head_dim = original_attention.head_dim
        # self.hidden_size = original_attention.hidden_size
        self.attention_dropout = original_attention.attention_dropout
        self.layer_idx = original_attention.layer_idx
        self.config = original_attention.config

        # Add your custom hook module
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Inline eager_attention_forward logic
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Pre-softmax hook
        attn_weights = self.attention_matrix_pre_softmax_hook(attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_matrix_hook(attn_weights)
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        # End inline

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights # type: ignore