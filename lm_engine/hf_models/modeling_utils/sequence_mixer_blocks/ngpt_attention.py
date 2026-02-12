# **************************************************
# nGPT: Normalized Transformer with Representation Learning on the Hypersphere
# Reference: https://arxiv.org/abs/2410.01131
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, is_torch_xla_available
from ...cache import GenerationCache
from ..chunk import contiguous_split
from ..position_embedding import apply_rotary_pos_emb
from .attention import Attention
from .utils import flash_attention


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


class nGPTAttention(Attention):
    """Attention with nGPT modifications:
    1. Q/K are L2-normalized per head after RoPE
    2. Learnable sqk scaling applied to normalized Q and K
    3. Inverted softmax scale: sqrt(head_dim) instead of 1/sqrt(head_dim)
    """

    def __init__(self, *args, **kwargs) -> nGPTAttention:
        super().__init__(*args, **kwargs)
        # nGPT uses inverted softmax scale
        self.attention_multiplier = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sqk: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
        accelerator = Accelerator.get_accelerator()

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None

            total_q = hidden_states.shape[0]
            input_shape = (total_q, self.num_key_value_heads, -1)
            output_shape = (total_q, -1, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]

            input_shape = (batch_size, query_length, self.num_key_value_heads, -1)
            output_shape = (batch_size, query_length, -1, self.head_dim)

        hidden_states = self.c_attn(hidden_states)
        hidden_states = hidden_states.view(*input_shape)

        query, key, value = (
            contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split
        )(
            hidden_states,
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim),
            dim=-1,
        )

        query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        # nGPT: L2-normalize Q and K per head, then apply sqk scaling
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        if sqk is not None:
            sqk_per_head = sqk.view(1, self.num_heads, 1, self.head_dim)
            query = query * sqk_per_head
            key = key * sqk_per_head

        if past_key_values is not None:
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

        if use_flash_attention_2 or use_flash_attention_3:
            assert accelerator == Accelerator.cuda

            if self.use_padding_free_transformer:
                output_shape = (-1, self.hidden_size)
            else:
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                output_shape = (batch_size, query_length, -1)

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            hidden_states = flash_attention(
                query=query,
                key=key,
                value=value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
                sliding_window=self.sliding_window,
            )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(*output_shape)
        else:
            assert self.sliding_window is None

            if accelerator == Accelerator.tpu:
                assert attention_mask is None
                assert self.softmax_dropout_p == 0

                hidden_states = flash_attention_tpu(
                    query,
                    key,
                    value,
                    causal=self.causal if attention_mask is None else False,
                    sm_scale=(
                        1 / math.sqrt(self.head_dim)
                        if self.attention_multiplier is None
                        else self.attention_multiplier
                    ),
                )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    is_causal=self.causal if attention_mask is None else False,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                )

            del query, key, value

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
