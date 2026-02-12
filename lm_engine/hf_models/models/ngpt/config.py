# **************************************************
# nGPT: Normalized Transformer with Representation Learning on the Hypersphere
# Reference: https://arxiv.org/abs/2410.01131
# **************************************************

from __future__ import annotations

import math

from ...config import CommonConfig


class nGPTConfig(CommonConfig):
    model_type = "ngpt"

    def __init__(
        self,
        ngpt_attn_alpha_init: float = 0.05,
        ngpt_mlp_alpha_init: float = 0.05,
        ngpt_alpha_init_scaling: float | None = None,
        ngpt_suv_init_scaling: float = 1.0,
        ngpt_sz_init_value: float = 1.0,
        ngpt_sz_init_scaling: float | None = None,
        **kwargs,
    ) -> nGPTConfig:
        # nGPT requires no weight tying
        kwargs["tie_word_embeddings"] = False

        self.ngpt_attn_alpha_init = ngpt_attn_alpha_init
        self.ngpt_mlp_alpha_init = ngpt_mlp_alpha_init
        self.ngpt_suv_init_scaling = ngpt_suv_init_scaling
        self.ngpt_sz_init_value = ngpt_sz_init_value

        super().__init__(**kwargs)

        # default init_scaling = 1 / sqrt(hidden_size) (the "base_scale" from the paper)
        if ngpt_alpha_init_scaling is None:
            ngpt_alpha_init_scaling = 1.0 / math.sqrt(self.hidden_size)
        self.ngpt_alpha_init_scaling = ngpt_alpha_init_scaling

        if ngpt_sz_init_scaling is None:
            ngpt_sz_init_scaling = 1.0 / math.sqrt(self.hidden_size)
        self.ngpt_sz_init_scaling = ngpt_sz_init_scaling
