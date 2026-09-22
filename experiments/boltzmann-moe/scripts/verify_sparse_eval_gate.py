#!/usr/bin/env python
"""Prove, by reading the gate on a LOADED model, that the sparse-eval fix does what we claim.

Loads the ORIGINAL unsharded dir and its sparseeval_* twin and prints `_sparse_active` for every
mixture module in each. Expected: False for the original (the bug), True for the twin (the fix).

Also exercises `activate_sparse_inference()` from the new
`lm_engine/hf_models/modeling_utils/mlp_blocks/sparse_eval.py` on the ORIGINAL model, which is the
in-memory route for callers that cannot restage a directory.

CPU-only: this reads a boolean attribute, it does not need a GPU and must not be timed.

  python scripts/verify_sparse_eval_gate.py <unsharded_dir> <sparseeval_dir>
"""
import sys, json
import torch
from transformers import AutoModelForCausalLM
from lm_engine.hf_models.modeling_utils.mlp_blocks.sparse_eval import (
    sparse_inference_report, activate_sparse_inference, assert_sparse_active)


def load(p):
    return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32,
                                                trust_remote_code=True).eval()


def show(tag, rep):
    print(f"\n--- {tag} ---")
    print(f"  {'module':28s} {'sp_fwd':>6s} {'sss':>5s} {'ACTIVE':>7s} {'K':>3s} {'k':>3s} {'p':>3s} {'r':>3s}")
    for r in rep:
        print(f"  {r['module']:28s} {str(r['sparse_forward']):>6s} {r['sparse_start_step']:>5d} "
              f"{str(r['active']):>7s} {r['n_experts']:>3d} {str(r['top_k']):>3s} "
              f"{str(r['candidates']):>3s} {r['proxy_rank']:>3d}")
    return rep


orig, sparse = sys.argv[1], sys.argv[2]
ro = show(f"ORIGINAL  {orig}", sparse_inference_report(load(orig)))
rs = show(f"SPARSEEVAL {sparse}", sparse_inference_report(load(sparse)))

print("\n--- activate_sparse_inference() on the ORIGINAL model (in-memory route) ---")
m = load(orig)
before = [r["active"] for r in sparse_inference_report(m)]
rep = activate_sparse_inference(m)
after = [r["now_active"] for r in rep]
print(f"  active before = {before}")
print(f"  active after  = {after}")
assert_sparse_active(m)
print("  assert_sparse_active(m) PASSED")

ok_bug = all(not r["active"] for r in ro if r["sparse_forward"])
ok_fix = all(r["active"] for r in rs if r["sparse_forward"])
ok_mem = all(after) and not any(before)
print("\n==================== VERDICT ====================")
print(f"  BUG  reproduced  (original runs DENSE oracle) : {ok_bug}")
print(f"  FIX  works       (sparseeval runs SPARSE)     : {ok_fix}")
print(f"  in-memory route works                          : {ok_mem}")
print("  ALL THREE CONFIRMED" if (ok_bug and ok_fix and ok_mem) else "  *** SOMETHING IS WRONG ***")
