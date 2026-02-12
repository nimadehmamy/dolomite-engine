# **************************************************
# nGPT: Normalized Transformer with Representation Learning on the Hypersphere
# Reference: https://arxiv.org/abs/2410.01131
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import nGPTConfig
from .layer import nGPTBlock


class nGPTPreTrainedModel(PreTrainedModelMixin):
    config_class = nGPTConfig
    layer_class = nGPTBlock
    _no_split_modules = ["nGPTBlock"]


class nGPTModel(nGPTPreTrainedModel, BaseModelMixin):
    def _init_model(self, config: nGPTConfig, **kwargs) -> None:
        super()._init_model(config, **kwargs)
        # nGPT has no final LayerNorm — hidden states are already on the hypersphere
        self.ln_f = nn.Identity()

    def _get_initial_hidden_state(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        hidden_state = super()._get_initial_hidden_state(input_ids, position_ids)
        # L2-normalize embeddings to place on the hypersphere
        return F.normalize(hidden_state, dim=-1)
