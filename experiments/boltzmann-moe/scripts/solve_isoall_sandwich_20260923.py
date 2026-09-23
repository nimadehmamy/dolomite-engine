"""Find a TRUE sandwich (1G, E x R, 1G) that matches a reference arm on TOTAL, ACTIVE and FLOPwt.

Why this exists: the 400M sandwich abl_Y is iso-TOTAL with the hybrid abl_X but 28.7% short on
ACTIVE and 27.5% short on FLOPwt, because [1,4,1] is 6 block applications against the hybrid's 12.
A sandwich has 2 dense blocks where the hybrid has 6, so the missing active parameters have to come
from somewhere: a wider dense block, a wider expert, or more recurrence.

The three axes respond differently, which is the whole difficulty:
  TOTAL   <- K * I_e   (every expert stored)
  ACTIVE  <- k * I_e   (only top_k run)
  FLOPwt  <- R * ACTIVE-of-the-energy-block
So raising I_e raises TOTAL K/k = 16x faster than it raises ACTIVE. Matching ACTIVE without
blowing TOTAL therefore needs the k/K RATIO to move, not just the width.

No parameter count is hand-derived here. This project's docs record that every hand-computed total
was LOW (swiglu c_fc is 2*I, energy_attention is 2*d*d not 4*d*d), so the affine coefficients in
(I_G, I_e) are MEASURED by calling audit_config at three points per (R,K,k) and differencing.
"""
import sys, copy, itertools, yaml
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config

REF  = "configs/iclr_26/ablations/abl_R_134M_hyb_rnorm_none.yml"
TMPL = "configs/iclr_26/ablations/abl_C_134M_1G1x6E1G_isototal.yml"

ref = audit_config(REF)
T_tot, T_act, T_flop = ref.total, ref.active, ref.flop_weight
print(f"reference {REF.split('/')[-1]}")
print(f"  TOTAL {T_tot/1e6:.2f}M  ACTIVE {T_act/1e6:.2f}M  FLOPwt {T_flop/1e6:.2f}M\n")

base = yaml.safe_load(open(TMPL))["model_args"]["pretrained_config"]
# locate the dense block and the mixture block in the template
di = next(i for i,b in enumerate(base["mlp_blocks"]) if b.get("mlp_type") == "MLP")
ei = next(i for i,b in enumerate(base["mlp_blocks"]) if b.get("mlp_type") == "EnergyFF_BoltzmannMoE")
print(f"template {TMPL.split('/')[-1]}: dense block idx {di}, energy block idx {ei}, "
      f"iters {base.get('layer_iterations')}\n")

def build(R, K, k, I_G, I_e):
    c = copy.deepcopy(base)
    c["layer_iterations"] = [1, R, 1]
    for i, b in enumerate(c["mlp_blocks"]):
        if b.get("mlp_type") == "MLP":
            b["intermediate_size"] = int(I_G)
        elif b.get("mlp_type") == "EnergyFF_BoltzmannMoE":
            b["n_experts"] = int(K); b["top_k"] = int(k)
            b["intermediate_size"] = int(K) * int(I_e)
    return c

def trip(R, K, k, I_G, I_e):
    a = audit_config(build(R, K, k, I_G, I_e))
    return a.total, a.active, a.flop_weight

# For fixed (R,K,k), each metric is affine in (I_G, I_e). Measure the gradients.
def solve_for(R, K, k):
    G0, E0, dG, dE = 2048, 1024, 512, 256
    b_tot, b_act, b_fl = trip(R, K, k, G0, E0)
    g_tot, g_act, g_fl = trip(R, K, k, G0 + dG, E0)
    e_tot, e_act, e_fl = trip(R, K, k, G0, E0 + dE)
    # d(metric)/d(I_G) and d(metric)/d(I_e)
    A = [[(g_act - b_act) / dG, (e_act - b_act) / dE],
         [(g_tot - b_tot) / dG, (e_tot - b_tot) / dE]]
    rhs = [T_act - b_act + A[0][0] * G0 + A[0][1] * E0,
           T_tot - b_tot + A[1][0] * G0 + A[1][1] * E0]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-9:
        return None
    I_G = (rhs[0] * A[1][1] - A[0][1] * rhs[1]) / det
    I_e = (A[0][0] * rhs[1] - rhs[0] * A[1][0]) / det
    return I_G, I_e

rows = []
for K, k in [(8,1),(8,2),(8,4),(16,1),(16,2),(16,4),(16,8),(32,2),(32,4),(32,8),(32,16),(4,1),(4,2)]:
    for R in range(3, 21):
        s = solve_for(R, K, k)
        if s is None: continue
        I_G, I_e = s
        if not (256 <= I_G <= 30000 and 64 <= I_e <= 30000): continue
        # round to a sane multiple and re-measure exactly
        I_Gr = int(round(I_G / 64) * 64); I_er = int(round(I_e / 32) * 32)
        if I_Gr <= 0 or I_er <= 0: continue
        tot, act, fl = trip(R, K, k, I_Gr, I_er)
        err = max(abs(tot-T_tot)/T_tot, abs(act-T_act)/T_act, abs(fl-T_flop)/T_flop) * 100
        rows.append((err, R, K, k, I_Gr, I_er, tot, act, fl))

rows.sort()
print(f"{'err%':>6s} {'R':>3s} {'K':>3s} {'k':>3s} {'I_G':>6s} {'I_G/d':>6s} {'I_e':>6s} {'I_e/d':>6s} "
      f"{'TOTAL':>8s} {'ACTIVE':>8s} {'FLOPwt':>8s} {'apps':>5s}")
print("-"*104)
d = base["hidden_size"]
for err, R, K, k, I_G, I_e, tot, act, fl in rows[:18]:
    print(f"{err:6.2f} {R:3d} {K:3d} {k:3d} {I_G:6d} {I_G/d:6.2f} {I_e:6d} {I_e/d:6.2f} "
          f"{tot/1e6:8.2f} {act/1e6:8.2f} {fl/1e6:8.2f} {R+2:5d}")
