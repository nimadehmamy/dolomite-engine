"""
Sparse INFERENCE activation -- eval/inference only, never imported by the training path.

THE BUG (found 2026-09-22)
--------------------------
`BoltzmannMoEFFEnergy.__init__` sets

    self._sparse_active = bool(sparse_forward) and self.sparse_start_step <= 0

and the ONLY thing that ever flips it is `set_training_step()`, which is called from exactly one
place in the tree: the training loop (`lm_engine/pretrain.py:403`). Nothing in `from_pretrained`,
`.eval()`, `.to()` or the eval harness touches it.

Consequence: every checkpoint TRAINED with `sparse_start_step > 0` -- which is all of them, because
the proxy must be distilled against exact all-K routing during a dense warmup phase -- reloads for
evaluation with `_sparse_active = False` and is therefore evaluated on the DENSE all-K path with
EXACT (oracle) routing. The proxy / surrogate router, whose approximation error IS the sparsity
claim, never ran at eval. `scripts/auto_eval_on_finish.sh:101` had already noticed the symptom
("reloads with _sparse_active FALSE, so eval runs the DENSE all-K path and single-block arms OOM at
batch 4") and worked around the resulting OOM with `batch_size 1` rather than the cause.

Two independent reasons this was invisible:
  * it fails SILENTLY and in the SAFE direction -- oracle routing is the best case, so nothing
    crashes and no number looks wrong; and
  * `energy_ff.py:952` explicitly says "the main use of sparse_forward is EVALUATING an
    already-trained checkpoint", so the intent was there and only the wiring was missing.

TWO WAYS TO FIX IT
------------------
1. CONFIG PATCH (preferred; zero code change, validated 2026-09-22 by job 1866475).
   Build a parallel eval directory whose `config.json` says `sparse_start_step: 0`, with the
   weights HARD LINKED so it costs no space and cannot drift:

       bash experiments/boltzmann-moe/scripts/make_sparse_eval_dir.sh <unsharded_dir> <dst>

   `sparse_start_step <= 0` makes `_sparse_active` True at construction and makes
   `set_training_step` a no-op, so the proxy loaded from the checkpoint is used verbatim and is
   never re-fitted. The ORIGINAL directory is untouched, which keeps the dense/oracle numbers
   available for the oracle-vs-sparse comparison the paper needs.

2. THIS MODULE, for code that already holds a live model object and cannot restage a directory
   (interactive analysis, a custom eval loop, a notebook).

WHY THIS IS A SEPARATE FILE
---------------------------
Making `_sparse_active` a persistent buffer, or flipping it inside `train(False)`, would change the
behaviour of the TRAINING path and of every existing checkpoint's strict load. `sparse_start_step`
must keep its training meaning: the dense warmup is load-bearing -- it is the phase in which the
proxy is distilled against the exact all-K routing distribution that the sparse path never
computes. So nothing here is imported by `energy_ff.py`, by the trainer, or by any config class.

DO NOT call `activate_sparse_inference` on a model you are about to train.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["activate_sparse_inference", "sparse_inference_report", "assert_sparse_active",
           "patch_config_for_sparse_eval"]


def _mixture_modules(model: Any):
    """Yield every module that owns the `_sparse_active` gate, with its qualified name."""
    for name, m in model.named_modules():
        if hasattr(m, "_sparse_active") and hasattr(m, "sparse_forward"):
            yield name, m


def activate_sparse_inference(model: Any, *, strict: bool = True) -> list[dict]:
    """Turn ON the proxy-sparse forward path for INFERENCE on an already-loaded model.

    Sets `_sparse_active = True` on every mixture module whose config asked for `sparse_forward`.
    Modules built WITHOUT `sparse_forward` are left alone -- there is no sparse path to take and
    forcing one would be a silent model change rather than a faithful evaluation.

    `_sparse_active` is a plain Python bool that dynamo guards on, so flipping it costs exactly one
    recompile -- the same single recompile the trainer pays at `sparse_start_step`.

    Returns one dict per mixture module describing what happened, so a caller can log or assert on
    it. Raises RuntimeError under `strict` if the model has no sparse-capable module at all, which
    almost always means the wrong checkpoint was loaded.
    """
    if getattr(model, "training", False):
        raise RuntimeError(
            "activate_sparse_inference() called on a model in TRAINING mode. The dense warmup "
            "phase governed by sparse_start_step is load-bearing (it distils the proxy against "
            "the exact all-K routing the sparse path never computes). Call model.eval() first, "
            "and never use this module in the training loop."
        )
    report: list[dict] = []
    for name, m in _mixture_modules(model):
        wanted = bool(getattr(m, "sparse_forward", False))
        before = bool(getattr(m, "_sparse_active"))
        if wanted:
            m._sparse_active = True
        report.append({
            "module": name,
            "sparse_forward": wanted,
            "sparse_start_step": int(getattr(m, "sparse_start_step", 0) or 0),
            "was_active": before,
            "now_active": bool(getattr(m, "_sparse_active")),
            "n_experts": int(getattr(m, "n_experts", 0) or 0),
            "top_k": getattr(m, "top_k", None),
            "candidates": getattr(m, "sparse_candidates", None),
            "proxy_rank": int(getattr(m, "proxy_rank", 0) or 0),
        })
    if not report:
        msg = "no sparse-capable mixture module found (nothing with both _sparse_active and sparse_forward)"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    flipped = [r for r in report if r["now_active"] and not r["was_active"]]
    logger.info("activate_sparse_inference: %d/%d mixture module(s) switched to the sparse path",
                len(flipped), len(report))
    return report


def sparse_inference_report(model: Any) -> list[dict]:
    """Read the gate WITHOUT changing it -- for auditing what an eval actually ran."""
    return [{
        "module": name,
        "sparse_forward": bool(getattr(m, "sparse_forward", False)),
        "sparse_start_step": int(getattr(m, "sparse_start_step", 0) or 0),
        "active": bool(getattr(m, "_sparse_active")),
        "candidates": getattr(m, "sparse_candidates", None),
        "top_k": getattr(m, "top_k", None),
        "n_experts": int(getattr(m, "n_experts", 0) or 0),
        "proxy_rank": int(getattr(m, "proxy_rank", 0) or 0),
    } for name, m in _mixture_modules(model)]


def assert_sparse_active(model: Any) -> None:
    """Fail loudly if any module that CAN run sparse is not running sparse.

    Put this after model construction in any eval that intends to measure the sparse path, so a
    dense-by-accident eval can never again be mistaken for a sparse one.
    """
    bad = [r["module"] for r in sparse_inference_report(model) if r["sparse_forward"] and not r["active"]]
    if bad:
        raise AssertionError(
            "these modules have sparse_forward but are running the DENSE all-K oracle path: "
            f"{bad}. Either set sparse_start_step: 0 in the checkpoint's config.json (see "
            "make_sparse_eval_dir.sh) or call activate_sparse_inference(model)."
        )


def patch_config_for_sparse_eval(config: dict) -> tuple[dict, int]:
    """Rewrite an HF `config.json` dict so the model reloads with the sparse path ACTIVE.

    Returns `(config, n_patched)`. Only touches mixture blocks that actually asked for
    `sparse_forward`, and only when their `sparse_start_step` is positive. This is the in-python
    twin of `make_sparse_eval_dir.sh`; keep the two in step.
    """
    n = 0
    for block in (config.get("mlp_blocks") or []):
        if block.get("sparse_forward") and int(block.get("sparse_start_step") or 0) > 0:
            block["sparse_start_step"] = 0
            n += 1
    return config, n
