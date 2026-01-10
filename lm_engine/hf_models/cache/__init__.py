# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Iterable

import torch

from ..config import CommonConfig
from .attention import _SoftmaxAttentionCache
from .mamba2 import _Mamba2Cache
from .rnn import _RNNCache


_CACHE_CLASSES = {
    "causal_convolution": _RNNCache,
    "gru": _RNNCache,
    "mamba2": _Mamba2Cache,
    "multihead_latent_attention": _SoftmaxAttentionCache,
    "rnn": _RNNCache,
    "softmax_attention": _SoftmaxAttentionCache,
}

CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None


class GenerationCache:
    def __init__(self, config: CommonConfig, **kwargs) -> "GenerationCache":
        """
        Build a cache entry for each *effective* layer (pre, repeated loop layers, post).
        Each effective layer maps to a base block index in the model, so we instantiate
        the cache class according to the base block's sequence_mixer_type.
        """
        # base indices for model blocks
        num_pre = config.num_pre_layers
        num_post = config.num_post_layers
        num_iter = getattr(config, "num_iterations", 1)
        num_layers = config.num_layers

        # calculate loop (the indices of shared blocks that are repeated)
        loop_base_idxs = list(range(num_pre, num_layers - num_post))
        pre_base_idxs = list(range(0, num_pre))
        post_base_idxs = list(range(num_layers - num_post, num_layers))

        # build effective order: pre, repeated loops, post
        effective_base_idxs: list[int] = []
        effective_base_idxs.extend(pre_base_idxs)
        for _ in range(num_iter):
            effective_base_idxs.extend(loop_base_idxs)
        effective_base_idxs.extend(post_base_idxs)

        # store mapping so we can inspect later if needed
        self.effective_base_idxs = effective_base_idxs

        # instantiate cache objects per effective layer using the base idx's type
        self.cache: list = [
            _CACHE_CLASSES[config.sequence_mixer_blocks[base_idx].sequence_mixer_type](config, base_idx, **kwargs)
            for base_idx in effective_base_idxs
        ]

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for c in self.cache:
            yield c.get_cache()

    def update(self, *, layer_idx: int, **kwargs) -> CACHE_TYPE:
        # layer_idx is expected to be the effective/unrolled layer id you pass from model
        return self.cache[layer_idx].update(**kwargs)

    def get_cache(self, layer_idx: int) -> CACHE_TYPE:
        print(len(self.cache))
        return self.cache[layer_idx].get_cache()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache[layer_idx].get_seq_length()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for cache in self.cache:
            cache.reorder_cache(beam_idx)


# class GenerationCache:
#     def __init__(self, config: CommonConfig, **kwargs) -> GenerationCache:
#         self.cache: list[_SoftmaxAttentionCache] = [
#             _CACHE_CLASSES[config.sequence_mixer_blocks[i].sequence_mixer_type](config, i, **kwargs)
#             for i in range(config.num_layers)
#         ]

#     def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
#         return self.cache[layer_idx].get_cache()

#     def __iter__(self) -> Iterable[CACHE_TYPE]:
#         for layer_idx in range(len(self)):
#             yield self.cache[layer_idx].get_cache()

#     def update(self, *, layer_idx: int, **kwargs) -> CACHE_TYPE:
#         return self.cache[layer_idx].update(**kwargs)

#     # TODO remove this function
#     def get_cache(self, layer_idx: int) -> CACHE_TYPE:
#         return self.cache[layer_idx].get_cache()

#     def get_seq_length(self, layer_idx: int = 0) -> int:
#         return self.cache[layer_idx].get_seq_length()

#     def reorder_cache(self, beam_idx: torch.Tensor) -> None:
#         for cache in self.cache:
#             cache.reorder_cache(beam_idx)
