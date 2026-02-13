# **************************************************
# nGPT: Normalized Transformer with Representation Learning on the Hypersphere
# Reference: https://arxiv.org/abs/2410.01131
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...mixins import CausalLMModelMixin
from ...modeling_utils import ParameterizedLinear
from .base import nGPTModel, nGPTPreTrainedModel
from .config import nGPTConfig


class nGPTForCausalLM(nGPTPreTrainedModel, CausalLMModelMixin):
    _tied_weights_keys = ["lm_head.weight"]
    base_model_class = nGPTModel

    def _init_model(self, config: nGPTConfig, **kwargs) -> None:
        super()._init_model(config, **kwargs)

        # sz: output logit scaling, one per vocab entry
        self.sz = nn.Parameter(config.ngpt_sz_init_scaling * torch.ones(config.vocab_size))
        self._sz_init_value = config.ngpt_sz_init_value
        self._sz_init_scaling = config.ngpt_sz_init_scaling

        # Normalize all weight matrices to start on the hypersphere
        self.normalize_all_weights()

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = (
            F.linear(hidden_states, self.transformer.wte.weight)
            if self._tied_word_embeddings
            else self.lm_head(hidden_states)
        )
        # Apply sz scaling: effective_sz = sz * (init_value / init_scaling)
        effective_sz = self.sz * (self._sz_init_value / self._sz_init_scaling)
        logits = logits * effective_sz
        return logits

    @torch.no_grad()
    def normalize_all_weights(self) -> None:
        """Normalize all weight matrices to unit norm.

        Called at initialization (before optimizer creation) AND after every optimizer step,
        as required by nGPT. Both the model parameters and the optimizer's parameter references
        must point to normalized weights (see Section 3.2 of the paper).

        Convention from the paper:
        - Input projections (embeddings, Q/K/V, MLP up-proj, LM head): normalize along dim=1
        - Output projections (attn c_proj, MLP c_proj): normalize along dim=0

        Note: Normalization is done in float32 for numerical stability (important for bf16/fp16
        training), then cast back to the original dtype.

        Note: dim=0 normalization requires the full (unsharded) tensor. With FSDP sharding
        (stage > 0), this method should be called inside a summon_full_params context or
        adapted for shard-aware normalization.
        """
        # Embedding weights: each embedding vector is unit norm
        _normalize_weight(self.transformer.wte.weight, dim=1)

        # LM head weights: each row is unit norm (input projection)
        # When tied, lm_head shares wte's weight — already normalized above
        if not self._tied_word_embeddings:
            _normalize_weight(self.lm_head.weight, dim=1)

        # Per-layer weights
        for block in self.transformer.h:
            # Attention input projection (c_attn): normalize along dim=1
            _normalize_weight(block.sequence_mixer.c_attn.weight, dim=1)
            # Attention output projection (c_proj): normalize along dim=0
            _normalize_weight(block.sequence_mixer.c_proj.weight, dim=0)
            # MLP input projection (c_fc): normalize along dim=1
            _normalize_weight(block.mlp_block.c_fc.weight, dim=1)
            # MLP output projection (c_proj): normalize along dim=0
            _normalize_weight(block.mlp_block.c_proj.weight, dim=0)


def _normalize_weight(weight: torch.nn.Parameter, dim: int) -> None:
    """Normalize a weight tensor in-place, casting to float32 for numerical stability.

    Uses manual norm + division instead of F.normalize to avoid clamp_min,
    which is not supported by DTensor (FSDP2).
    """
    dtype = weight.data.dtype
    w = weight.data.float()
    weight.data.copy_((w / w.norm(p=2, dim=dim, keepdim=True)).to(dtype=dtype))
