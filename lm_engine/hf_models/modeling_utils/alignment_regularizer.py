"""Cross-layer alignment regularizer for energy / Boltzmann MoE models.

Encourages layer outputs (and/or expert weights) to stay aligned across
consecutive energy layers. The hypothesis is that alignment indicates
productive iteration -- layers that drift apart are wasting recurrence budget.

Integrates through ``add_aux_loss()`` so the gradient flows through the same
mechanism as repulsion, proxy KL, and Sinkhorn balancing -- no model-wrapper
changes needed.

Config fields (all on pretrained_config, read via ``getattr``):
    align_coef      : float = 0.0     # 0 = off
    align_mode      : str   = "output" | "weight" | "both"
    align_pairs     : str   = "consecutive" | "all" | "random_skip"
    align_subsample : int   = 64      # row subsample for weight alignment
    align_ramp_steps: int   = 0       # cosine ramp for coefficient (0 = no ramp)

Usage -- call once per forward pass in the base model (energy/base.py),
after all blocks have run:

    from ...modeling_utils.alignment_regularizer import apply_alignment_loss
    apply_alignment_loss(self, global_step=self._some_step_counter)

Or from the model wrapper (pretraining.py) after model(...) returns.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from ..loss import add_aux_loss


# Mixer types that count as "energy" blocks for alignment purposes.
_ENERGY_MIXER_TYPES = frozenset({
    "energy_attention",
    "mixed_head_energy_descent",
    "boltzmann_moe_energy_attention",
    "boltzmann_moe_paired_unit",
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cosine_ramp(base: float, ramp_steps: int, step: int) -> float:
    """Cosine ramp: 0 -> base over ramp_steps, constant after."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return base
    return base * (1.0 - math.cos(math.pi * step / ramp_steps)) / 2.0


def _get_energy_blocks(model) -> list[tuple[int, torch.nn.Module]]:
    """Return (idx, block) for every energy/Boltzmann block in the model."""
    # Navigate FSDP / model wrappers to reach the layer list.
    # Handles: BaseModelMixin.h, CausalLM.transformer.h, wrapper.model.transformer.h
    blocks = None
    for path in (
        lambda m: m.h,                          # BaseModelMixin (called from within forward)
        lambda m: m.transformer.h,              # CausalLM level
        lambda m: m.model.transformer.h,        # FSDP wrapper
        lambda m: m.module.model.transformer.h, # double wrapper
    ):
        try:
            blocks = path(model)
            break
        except AttributeError:
            continue
    if blocks is None:
        return []

    result = []
    for idx, blk in enumerate(blocks):
        mixer = getattr(blk, "sequence_mixer_type", None)
        if mixer in _ENERGY_MIXER_TYPES:
            result.append((idx, blk))
    return result


def _select_pairs(
    n: int,
    mode: str,
    k_sample: int = 3,
) -> list[tuple[int, int]]:
    """Return index pairs (into the energy-block list) to regularise.

    Args:
        n:        number of energy blocks.
        mode:     "consecutive" | "all" | "random_skip"
        k_sample: for "random_skip", how many pairs to sample.
    """
    if n < 2:
        return []

    if mode == "consecutive":
        return [(i, i + 1) for i in range(n - 1)]

    if mode == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    if mode == "random_skip":
        # Sample k_sample pairs uniformly from all possible pairs.
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = min(k_sample, len(all_pairs))
        return random.sample(all_pairs, k)

    raise ValueError(f"unknown align_pairs mode: {mode!r}")


# ---------------------------------------------------------------------------
# output alignment
# ---------------------------------------------------------------------------

def _output_alignment_loss(
    blocks: list[tuple[int, torch.nn.Module]],
    pairs: list[tuple[int, int]],
) -> torch.Tensor | None:
    """1 - cos_sim(attn_out_i, attn_out_j) averaged over pairs and tokens.

    Reads ``block._attn_out`` which EnergyBlock.forward stores on every
    forward pass for energy blocks.  No extra forward needed.
    """
    outs = []
    for _, blk in blocks:
        out = getattr(blk, "_attn_out", None)
        if out is None:
            return None
        outs.append(out)

    terms = []
    for i, j in pairs:
        if i >= len(outs) or j >= len(outs):
            continue
        flat_i = outs[i].reshape(-1, outs[i].shape[-1])
        flat_j = outs[j].reshape(-1, outs[j].shape[-1])
        cs = F.cosine_similarity(flat_i, flat_j, dim=-1)  # (B*T,)
        terms.append((1.0 - cs).mean())

    if not terms:
        return None
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# weight alignment
# ---------------------------------------------------------------------------

def _weight_alignment_loss(
    blocks: list[tuple[int, torch.nn.Module]],
    pairs: list[tuple[int, int]],
    subsample: int = 64,
) -> torch.Tensor | None:
    """Cosine distance between corresponding expert weight matrices.

    For each pair of energy blocks, compares the fused W1 and W2 weight
    matrices of their Boltzmann MoE layers. Uses random row subsampling
    to keep cost O(subsample * hidden_size) per pair.

    Falls back to ``_weight_named_flat`` (the per-matrix flat vectors stored
    by EnergyBlock.forward) if the block does not have a fused expert holder.
    """
    terms = []
    for bi, bj in pairs:
        if bi >= len(blocks) or bj >= len(blocks):
            continue
        _, blk_i = blocks[bi]
        _, blk_j = blocks[bj]

        # Try fused expert holder first (Boltzmann MoE w1w2)
        holder_i = _get_expert_holder(blk_i)
        holder_j = _get_expert_holder(blk_j)

        if holder_i is not None and holder_j is not None:
            for w_attr in ("W1", "W2"):
                Wi = getattr(holder_i, w_attr, None)
                Wj = getattr(holder_j, w_attr, None)
                if Wi is None or Wj is None:
                    continue
                Wi_w = Wi.weight  # (K*I_e, hidden)
                Wj_w = Wj.weight

                n_rows = Wi_w.shape[0]
                if subsample > 0 and subsample < n_rows:
                    idx = torch.randint(n_rows, (subsample,), device=Wi_w.device)
                    Wi_sub = Wi_w[idx]
                    Wj_sub = Wj_w[idx]
                else:
                    Wi_sub = Wi_w
                    Wj_sub = Wj_w

                cs = F.cosine_similarity(
                    Wi_sub.reshape(1, -1), Wj_sub.reshape(1, -1)
                )
                terms.append((1.0 - cs).mean())
        else:
            # Fallback: use _weight_named_flat stored by EnergyBlock.forward
            nf_i = getattr(blk_i, "_weight_named_flat", None)
            nf_j = getattr(blk_j, "_weight_named_flat", None)
            if nf_i is not None and nf_j is not None:
                shared = set(nf_i.keys()) & set(nf_j.keys())
                for name in sorted(shared):
                    vi, vj = nf_i[name], nf_j[name]
                    if subsample > 0 and subsample < vi.numel():
                        idx = torch.randint(vi.numel(), (subsample,), device=vi.device)
                        vi, vj = vi[idx], vj[idx]
                    cs = F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0))
                    terms.append((1.0 - cs).mean())

    if not terms:
        return None
    return torch.stack(terms).mean()


def _get_expert_holder(block) -> torch.nn.Module | None:
    """Navigate to the fused expert holder inside a BoltzmannMoE block."""
    ffwd = getattr(block, "ffwd", None)
    if ffwd is None:
        return None
    # FusedMoEContainer stores the holder at .expert_holder
    holder = getattr(ffwd, "expert_holder", None)
    if holder is not None:
        return holder
    # Legacy path: holder might be at .holder
    holder = getattr(ffwd, "holder", None)
    return holder


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def apply_alignment_loss(
    model,
    global_step: int = 0,
) -> None:
    """Compute and add alignment regularisation to the global aux loss.

    Reads config from ``model.config`` (or nearest wrapper):
        align_coef, align_mode, align_pairs, align_subsample, align_ramp_steps

    Fires ``add_aux_loss()`` so the gradient merges with repulsion / proxy KL.
    """
    # Navigate to config
    cfg = None
    for obj in (model, getattr(model, "model", None), getattr(model, "module", None)):
        if obj is not None and hasattr(obj, "config"):
            cfg = obj.config
            break
    if cfg is None:
        return

    base_coef = float(getattr(cfg, "align_coef", 0.0) or 0.0)
    if base_coef <= 0:
        return

    ramp = int(getattr(cfg, "align_ramp_steps", 0) or 0)
    coef = _cosine_ramp(base_coef, ramp, global_step)
    if coef <= 0:
        return

    mode = str(getattr(cfg, "align_mode", "output"))
    pair_mode = str(getattr(cfg, "align_pairs", "consecutive"))
    subsample = int(getattr(cfg, "align_subsample", 64))

    blocks = _get_energy_blocks(model)
    if len(blocks) < 2:
        return

    pairs = _select_pairs(len(blocks), pair_mode)
    if not pairs:
        return

    total = None

    if mode in ("output", "both"):
        out_loss = _output_alignment_loss(blocks, pairs)
        if out_loss is not None:
            total = out_loss if total is None else total + out_loss

    if mode in ("weight", "both"):
        w_loss = _weight_alignment_loss(blocks, pairs, subsample=subsample)
        if w_loss is not None:
            total = w_loss if total is None else total + w_loss

    if total is not None:
        add_aux_loss(coef * total)
