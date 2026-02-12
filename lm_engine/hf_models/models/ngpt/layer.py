# **************************************************
# nGPT: Normalized Transformer with Representation Learning on the Hypersphere
# Reference: https://arxiv.org/abs/2410.01131
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache import GenerationCache
from ...modeling_utils import get_mlp_block, get_sequence_mixer
from ...modeling_utils.activations import is_glu
from .config import nGPTConfig


class nGPTBlock(nn.Module):
    def __init__(
        self, config: nGPTConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> nGPTBlock:
        super().__init__()

        hidden_size = config.hidden_size
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type

        # Reuse standard sequence mixer and MLP from factory
        self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
        self.mlp_block = get_mlp_block(config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx)

        # nGPT learnable parameters
        alpha_init_scaling = config.ngpt_alpha_init_scaling

        # sqk: Q/K scaling after normalization (effective init value = 1.0)
        self.sqk = nn.Parameter(alpha_init_scaling * torch.ones(hidden_size))
        self._sqk_init_value = 1.0
        self._sqk_init_scaling = alpha_init_scaling

        # attn_alpha: attention SLERP step size (effective init value = ngpt_attn_alpha_init)
        self.attn_alpha = nn.Parameter(alpha_init_scaling * torch.ones(hidden_size))
        self._attn_alpha_init_value = config.ngpt_attn_alpha_init
        self._attn_alpha_init_scaling = alpha_init_scaling

        # mlp_alpha: MLP SLERP step size (effective init value = ngpt_mlp_alpha_init)
        self.mlp_alpha = nn.Parameter(alpha_init_scaling * torch.ones(hidden_size))
        self._mlp_alpha_init_value = config.ngpt_mlp_alpha_init
        self._mlp_alpha_init_scaling = alpha_init_scaling

        # suv: SwiGLU/GLU rescaling (effective init value = sqrt(hidden_size))
        mlp_block_config = config.mlp_blocks[layer_idx]
        if is_glu(mlp_block_config.activation_function):
            suv_size = 2 * mlp_block_config.intermediate_size
        else:
            suv_size = mlp_block_config.intermediate_size
        self.suv = nn.Parameter(config.ngpt_suv_init_scaling * torch.ones(suv_size))
        self._suv_init_value = math.sqrt(hidden_size)
        self._suv_init_scaling = config.ngpt_suv_init_scaling

    def _get_effective_param(self, param: torch.Tensor, init_value: float, init_scaling: float) -> torch.Tensor:
        return param * (init_value / init_scaling)

    def _slerp_residual(
        self, h: torch.Tensor, sublayer_output: torch.Tensor, alpha: torch.Tensor, init_value: float, init_scaling: float
    ) -> torch.Tensor:
        """nGPT SLERP-like residual connection on the hypersphere."""
        lr = torch.abs(self._get_effective_param(alpha, init_value, init_scaling))
        A_norm = F.normalize(h, dim=-1)
        B_norm = F.normalize(sublayer_output, dim=-1)
        res = A_norm + lr * (B_norm - A_norm)
        return F.normalize(res, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # --- Attention sub-layer ---
        # Compute effective sqk for Q/K scaling
        effective_sqk = self._get_effective_param(self.sqk, self._sqk_init_value, self._sqk_init_scaling)

        attn_output = self._sequence_mixer_forward(
            hidden_states=hidden_states,
            sqk=effective_sqk,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # SLERP residual for attention
        hidden_states = self._slerp_residual(
            hidden_states, attn_output, self.attn_alpha, self._attn_alpha_init_value, self._attn_alpha_init_scaling
        )

        # --- MLP sub-layer with suv rescaling ---
        # We reach into MLP submodules to insert suv scaling between c_fc and activation
        x = self.mlp_block.c_fc(hidden_states)
        effective_suv = self._get_effective_param(self.suv, self._suv_init_value, self._suv_init_scaling)
        x = x * effective_suv
        x = self.mlp_block.act(x)
        x = self.mlp_block.c_proj(x)
        x = self.mlp_block.dropout(x)

        # SLERP residual for MLP
        hidden_states = self._slerp_residual(
            hidden_states, x, self.mlp_alpha, self._mlp_alpha_init_value, self._mlp_alpha_init_scaling
        )

        return hidden_states

    def _sequence_mixer_forward(
        self,
        hidden_states: torch.Tensor,
        sqk: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.sequence_mixer_type == "ngpt_softmax_attention":
            hidden_states = self.sequence_mixer(
                hidden_states,
                sqk=sqk,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type in ["softmax_attention", "multihead_latent_attention"]:
            hidden_states = self.sequence_mixer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type in ["causal_convolution", "mamba2"]:
            hidden_states = self.sequence_mixer(
                hidden_states, cache_params=past_key_values, attention_mask=attention_mask
            )
        elif self.sequence_mixer_type in ["gru", "rnn"]:
            hidden_states = self.sequence_mixer(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type == "gated_deltanet":
            hidden_states = self.sequence_mixer(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({self.sequence_mixer_type})")

        return hidden_states
