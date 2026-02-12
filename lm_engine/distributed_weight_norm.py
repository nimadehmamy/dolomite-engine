# **************************************************
# nGPT weight normalization with FSDP support
# **************************************************

from __future__ import annotations

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _get_hf_model(model: nn.Module) -> nn.Module | None:
    """Extract the underlying HuggingFace model from the wrapper stack.

    The model hierarchy is:
    - FSDP1: FSDP(ModelWrapper) -> .model -> nGPTForCausalLM
    - FSDP2/DDP: ModelWrapper -> .model -> nGPTForCausalLM
    - torch.compile: OptimizedModule(ModelWrapper) -> .model -> nGPTForCausalLM
    """
    # Navigate through wrapper layers to find the HF model
    hf_model = getattr(model, "model", None)
    if hf_model is not None and hasattr(hf_model, "normalize_all_weights"):
        return hf_model
    return None


def maybe_normalize_ngpt_weights(model: nn.Module, fsdp_algorithm: int) -> None:
    """Normalize nGPT weight matrices after optimizer step, with FSDP support.

    Args:
        model: The (possibly wrapped) model from model_container
        fsdp_algorithm: 1 for FSDP1, 2 for FSDP2/DDP
    """
    hf_model = _get_hf_model(model)
    if hf_model is None:
        return

    if fsdp_algorithm == 1 and isinstance(model, FSDP):
        # FSDP1: need to unshard parameters for correct dim=0 normalization
        with FSDP.summon_full_params(model, writeback=True):
            hf_model.normalize_all_weights()
    else:
        # FSDP2 (DTensor handles distributed ops) / DDP / single GPU
        hf_model.normalize_all_weights()
