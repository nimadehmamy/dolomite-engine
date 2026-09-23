# The `hopfield_grad_scale` prefactor ladder (abl_J / abl_K / abl_L)

Created 2026-09-21. **Not yet launched.** Four arms, all on the pure `1x12E` 134M
sparse(proxy) hopfield base, differing ONLY in how the FF branch's returned vector relates to
`grad_h E`. Params (TOTAL / ACTIVE / FLOPwt) are **identical across all four** — the knob is a
scalar prefactor with no parameters, so this is the cleanest ablation in the project.

## What the knob actually is

The hopfield expert's energy is `E = mean_j gelu(Wh)_j^2`, so
`grad_h E = (2/I_e) W^T (gelu(Wh) . gelu'(Wh))`.

`_gelu_and_grad` does NOT return the true `gelu'` for the default method: `"sigmoid"` returns
`phi' = 0.5*sigmoid(1.702 x)`, i.e. HALF the derivative (and only an approximation of its shape).
`"erf_exact"` returns the true derivative. **So `hopfield_grad_scale` and `gelu_grad_method` are
NOT independent** — a mode that is exact with one is off by 2x with the other. Verified by
autograd in float64 (`||forward|| / ||dE/dh||`, and the cosine between them):

| grad_scale | gelu_grad_method | ratio | cos | verdict |
|---|---|---|---|---|
| `exact` | `erf_exact` | **1.000000** | **1.00000000** | EXACT energy descent |
| `exact` | `tanh_exact` | 1.000467 | 0.99999979 | exact to 5e-4 |
| `exact` | `sigmoid` | 0.510497 | 0.99682862 | HALF — do not pair |
| `mean` | `sigmoid` | 1.020995 | 0.99682862 | right magnitude, approx direction |
| `mean` | `erf_exact` | 2.000000 | 1.00000000 | 2x too large — do not pair |
| `sqrt_consistent` | `erf_exact` | 16.0 (at I=64) | 1.00000000 | = sqrt(I_e) too large |

`exact` (= `2/I_e`) was ADDED 2026-09-21. It is the only mode that returns `grad_h E`.

## Why the knob exists at all, and why only for hopfield

It is not a correctness fix — it deliberately BREAKS the gradient identity to cure a magnitude
problem (the FF branch measured inert, 0.04% of `grad_E`). The reason hopfield needs it and
**w1w2 does not** is structural:

* hopfield `E = (1/I_e) sum_j gelu(w_j h)^2` — a sum of `I_e` **positive** terms. The sum is
  O(I_e), so `1/I_e` is forced to keep `E = O(1)`; but the gradient is a sum of `I_e`
  **incoherent vectors**, so it is only O(sqrt(I_e)), leaving `grad E ~ 1/sqrt(I_e)`.
  **You cannot have `E = O(1)` and `grad E = O(1)` simultaneously with a positive-definite
  energy of this form.** That is the whole tension, and it is intrinsic, not a bug.
* w1w2 `E = -(1/sqrt(I_e)) sum_j phi(W1h)_j (W2h)_j` — a sum of `I_e` **signed** terms, so the
  sum is already O(sqrt(I_e)). One `1/sqrt(I_e)`, living in the ENERGY definition, normalises
  `E` AND the gradient at once. `W1W2FFEnergy.forward` returns `-grad_h E` exactly, with no
  knob. This is the H-series' normalisation, and it is exact by construction.

So `abl_L` is the hopfield analogue of what the H-series/w1w2 line does: exact descent on a
properly normalised energy. The shipped `sqrt_consistent` is 66.9x that at `I_e = 4480`.

Call sites are hopfield-only: `HopfieldFFEnergy.forward`, `_HopfieldExpert.forward`, and
`BoltzmannMoEFFEnergy._forward_fused` / `_forward_sparse` (which wrap hopfield experts).

## The four arms (I_e = 4480, so sqrt(I_e) = 66.93)

| arm | grad_scale | gelu | prefactor | x exact grad | status |
|---|---|---|---|---|---|
| `cmix_134M_pure_32B_sparse` (reference) | `sqrt_consistent` | sigmoid | 0.05976143 | **66.93** | **DONE, 32B, Avg11 42.00 / MMLU 24.43** |
| `abl_K_134M_pure_1x12E_gsInvSqrt` | `inv_sqrt` | sigmoid | 0.01494036 | 16.73 | config ready |
| `abl_J_134M_pure_1x12E_gsMean` | `mean` | sigmoid | 0.00089286 | 1.00 (cos 0.997) | config ready |
| `abl_L_134M_pure_1x12E_gsExact` | `exact` | **erf_exact** | 0.00044643 | **1.000 (cos 1.000)** | config ready |

Two questions are separated by this design:
* **magnitude** — 1x (J/L) vs 16.7x (K) vs 66.9x (reference);
* **direction fidelity at fixed magnitude** — J (cos 0.9968, sigmoid approx) vs L (cos 1.0).

## Reference curve — no need to re-run the sqrt_consistent side

From the finished reference's logs (`lm_loss`, nats, lower better; 262,144 tok/step):

| tokens | step | lm_loss |
|---|---|---|
| 0.52B | 2,000 | 4.3486 |
| 1.31B | 5,000 | 3.7345 |
| 2.62B | 10,000 | 3.4915 |
| **4.00B** | **15,260** | **3.4180** |
| 5.24B | 20,000 | 3.4043 |
| **8.00B** | **30,520** | **3.3344** |

**Zero grad_norm excursions above 5.0 across all 12,207 logged steps of the full 32B run** — so
`sqrt_consistent` was NOT unstable at this scale, which is evidence against the hypothesis that
the 1/sqrt form is destabilising. The reference also has `milestones/unsharded_tok8B_step32000`,
so an 8B Avg11 comparison point needs only a 1-GPU eval, no retraining.

## Priors

`mean` is the pre-2026-09-12 default under which the branch was measured inert, so the expected
outcome is that J loses. `abl_L` is the interesting one: it is the only arm that is honestly
"energy descent", and if it matches the reference then the 66.9x inflation buys nothing and the
paper can claim exact descent.

## Cost

0.521 s/step on 8 GPUs (measured on the reference) => 4B in **2.2 h**, 8B in **4.4 h**, 32B in
17.7 h, per arm. Schedule is the parent's: 2000 warmup + 0 constant + 120,070 cosine decay —
**pure cosine, not WSD**. Identical across all four, so the A/B is clean, but it is not the WSD
shape the 400M arms use.

## Micro-batch size at 1 GPU: mbs 2 is the ONLY shape that fits (measured 2026-09-21)

Tested because `mbs x sequence_length` is the per-call token count that decides whether
`sparse_forward` wins (S12.9: it LOSES at 4,096 tokens/call, WINS 2.1-2.4x at 16,384). At 1 GPU,
mbs 2 gives 8,192 tokens/call and mbs 4 would give 16,384 -- the winning regime. Both
alternatives were submitted iso-token (mbs x ga held at 64, so 262,144 tok/step in every case):

| mbs | ga | tok/call | result |
|---|---|---|---|
| 2 | 32 | 8,192 | **RUNS** |
| 4 | 16 | 16,384 | **OOM** -- 77.20 GiB in use of 79.18 GiB, died allocating 2.19 GiB more |
| 8 | 8 | 32,768 | **OOM** |

The reason is the absence of FSDP sharding: the 8-GPU parent's own header records
`OOM FIX (2026-09-17): micro_batch_size 4 -> 2` *with* 8-way sharding, so at 1 GPU, where the
full model, gradients and optimizer states sit on one card, mbs 4 is far out of reach. **The
S12.9 sparse-speedup regime is therefore unreachable at 1 GPU** -- it needs the multi-GPU
config, where per-GPU memory is divided.

NOTE the dense phase: `sparse_start_step: 300`, so the first 300 steps evaluate ALL K=16 experts
rather than top-2 (~8x the MoE FLOPs). Step time fell 12.24 s (step 10) -> 6.61 s (step 20) as
the compile warmup cleared, and should drop again at step 300. **Do not extrapolate a schedule
from any step_time read before step 300** -- the config header says exactly this.
