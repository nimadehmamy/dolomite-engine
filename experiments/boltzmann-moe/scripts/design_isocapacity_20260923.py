#!/usr/bin/env python
"""Search for iso-capacity arms and AUDIT each candidate (never hand-compute params here).

Targets are the 134M hybrid `6G1x6E`: TOTAL 134.3M, ACTIVE 123.2M, FLOPwt 141.7M, I_e/hidden 1.33.
Both designs are pinned to hidden=768 so the tied embedding is 77.07M -- the same as every arm in the
134M tier -- which is the whole point: abl_P's h=1088 put 109.2M into the embedding and left only
22.8M of non-embedding capacity against the hybrid's 57.2M.

DESIGN A: the [2,2,3] structure at h=768 (3 distinct energy blocks, 7 applications).
DESIGN B: a PURE deep energy stack -- 6 distinct energy blocks, recurrence free to vary so that
          FLOPwt matches while ACTIVE matches.
Point 6 of the pre-flight is enforced: I_e/hidden >= 0.5, and we prefer ~1.33 to match the hybrid.
"""
import copy, itertools, os, sys, tempfile, yaml
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config

BASE = "configs/iclr_26/ablations/abl_P2_134M_pure_223_K4.yml"
T_TOT, T_ACT, T_FLOP = 134.25, 123.24, 141.72   # the hybrid, from audit_config
V = 100352

base = yaml.safe_load(open(BASE))

def build(hidden, iters, I_tot, K, k=2, heads=12):
    c = copy.deepcopy(base)
    pc = c["model_args"]["pretrained_config"]
    pc["hidden_size"] = hidden
    pc["num_layers"] = len(iters)
    pc["layer_iterations"] = list(iters)
    smb = pc["sequence_mixer_blocks"][0]
    proto_sm = {kk: vv for kk, vv in smb.items()}
    proto_sm["num_attention_heads"] = heads
    proto_sm["num_key_value_heads"] = heads
    proto_sm["sequence_mixer_type"] = "energy_attention"      # PURE: energy attention everywhere
    pc["sequence_mixer_blocks"] = [copy.deepcopy(proto_sm) for _ in iters]
    moe = None
    for b in pc["mlp_blocks"]:
        if b.get("n_experts"):
            moe = b; break
    out = []
    for i, n in enumerate(iters):
        b = copy.deepcopy(moe)
        b["n_experts"] = K
        b["top_k"] = k
        b["intermediate_size"] = I_tot
        b["sparse_candidates"] = k
        b["sinkhorn_mu_iters"] = int(n)     # MUST equal this block's layer_iterations entry
        out.append(b)
    pc["mlp_blocks"] = out
    return c

def audit(c):
    fd, p = tempfile.mkstemp(suffix=".yml"); os.close(fd)
    yaml.safe_dump(c, open(p, "w"), sort_keys=False)
    try:
        a = audit_config(p)
        return a.total/1e6, a.active/1e6, a.flop_weight/1e6
    finally:
        os.unlink(p)

def score(t, a, f):
    return max(abs(t-T_TOT)/T_TOT, abs(a-T_ACT)/T_ACT, abs(f-T_FLOP)/T_FLOP)

print(f"targets (134M hybrid): TOTAL {T_TOT} ACTIVE {T_ACT} FLOPwt {T_FLOP}; embed at h=768 = {V*768/1e6:.2f}M\n")

for label, iter_opts in [("DESIGN A  [2,2,3] h768", [(2,2,3)]),
                         ("DESIGN B  pure 6 blocks", [(1,1,1,1,1,1),(1,1,1,1,1,2),(1,1,1,1,2,2),
                                                      (1,1,1,2,2,2),(1,1,2,2,2,2),(1,1,1,1,1,3),
                                                      (2,2,2,2,2,2),(1,1,1,1,2,3)])]:
    print("="*100); print(label)
    print(f"  {'iters':16s} {'apps':>4s} {'K':>3s} {'I_tot':>6s} {'I_e':>5s} {'Ie/h':>5s} {'TOTAL':>8s} {'ACTIVE':>8s} {'FLOPwt':>8s} {'worst dev':>9s}")
    rows=[]
    for iters in iter_opts:
        for K in (4, 8, 16):
            for I_tot in range(2048, 40961, 1024):
                if I_tot % K: continue
                ie = I_tot//K
                if ie/768 < 0.5: continue
                try: t,a,f = audit(build(768, iters, I_tot, K))
                except Exception: continue
                rows.append((score(t,a,f), iters, sum(iters), K, I_tot, ie, t,a,f))
    rows.sort()
    for s,it,ap,K,I,ie,t,a,f in rows[:6]:
        print(f"  {str(it):16s} {ap:>4d} {K:>3d} {I:>6d} {ie:>5d} {ie/768:>5.2f} {t:>7.1f}M {a:>7.1f}M {f:>7.1f}M {100*s:>8.1f}%")
