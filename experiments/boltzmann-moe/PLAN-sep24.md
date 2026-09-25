# PLAN: FineWeb-Edu fast sweeps — isolate what broke Boltzmann MoE

**Date:** 2026-09-24 (started), updated 2026-09-25 02:40 UTC
**Deadline:** end of Sep 25, 2026
**Data:** nemotron-cc p2_0 (granite-4.0 tiktoken, proven megatron format)

## Motivation

Every new-era Boltzmann MoE arm trails its baseline. The HANDOFF's own verdict (§19.5):
*"Every variant that improved loss did so by REMOVING or REPLACING the Boltzmann mechanism."*
But the old h1_boltz_moe_580m (679M, legacy w1w2 dense, K=4) beat Switch by +0.65pp.
Five things changed simultaneously; we're isolating them.

---

## Round 1 RESULTS ✅ (d=768, cosine w/ constant phase, 4 GPUs each)

All `lm_loss` (nats, lower better). Avg11 = 11-task accuracy mean (pp, higher better).
MMLU reported separately. GSM8K skipped for speed.

| arm | code | K | k | total M | tokens | lm_loss | Avg11 | MMLU |
|---|---|---|---|---|---|---|---|---|
| **A** 12G baseline | MLP | — | — | 162 | 4.86B | **3.164** | **43.95** | 26.33 |
| **C** new w1w2 dense | EnergyFF | 4 | 4 | 133 | 4.00B | 3.236 | **43.96** | 24.06 |
| **B** legacy w1w2 dense | BoltzMoE_MLP | 4 | 4 | 133 | 4.00B | 3.220 | 43.62 | 23.42 |
| **D** new w1w2 sparse | Surrogate | 16 | 2 | 158 | 2.41B | 3.313 | 43.27 | 24.82 |

### Key findings — Round 1
1. **New code ≈ legacy code.** C (43.96) ≈ B (43.62), Δ=0.34pp. Within single-seed noise (~0.87pp).
   New code is actually marginally better on Avg11 despite higher lm_loss. **Safe to use new code.**
2. **Boltzmann MoE matches the 12G baseline on Avg11** (43.96 vs 43.95) despite 18% fewer params
   and 18% fewer tokens. The lm_loss gap (3.236 vs 3.164) did NOT translate to downstream.
3. **Sparse trailed by 0.69pp** (43.27 vs 43.96) — but got 40% fewer tokens (2.41B vs 4.0B).
   Confounded. Round 2 fixes this.
4. **Schedule bug:** used 83% constant + 10% decay (effectively WSD). Fixed to proper cosine in R2.

---

## Round 2 IN PROGRESS (d=768, proper cosine schedule, iso-total 162M, 4.0B tokens each)

All arms at 162M total params, 4.0B tokens, 7629 steps, cosine decay (500 warmup + 7129 decay).
4 GPUs each on grp_ebm.

| arm | job | K | k | I_e | s/step | step now | loss now | ETA |
|---|---|---|---|---|---|---|---|---|
| **E1** w1w2 dense K=8 | KILLED | 8 | 8 | 3361 | 6.6 | 520 | 4.59 | — |
| **E2** w1w2 sparse K=8 | 1907357 | 8 | 2 | 3361 | 4.5 | 1800 | 3.63 | ~09:30 |
| **E3** Switch K=8 | 1907614 | 8 | 2 | 2240 | 1.2 | 5600 | 3.27 | ~03:00 |
| **D2** w1w2 sparse K=16 iso-FLOP | 1907359 | 16 | 2 | 1921 | 3.2 | 2180 | 3.61 | ~07:30 |
| **E4** psd_anti probe (1B) | 1907360 | 8 | 8 | 3361 | 6.6 | 1140 | 3.83 | ~03:30 |
| **E5** unconstrained probe (1B) | 1907615 | 8 | 8 | 3361 | 6.7 | 1030 | 3.95 | ~04:00 |

### Emerging findings — Round 2
- **E2 ≈ D2** (3.63 vs 3.61 at comparable steps) — K=8 and K=16 sparse are indistinguishable.
  More experts at matched compute does not help (or hurt) at this scale.
- **E3 Switch leads on raw loss** (3.27 at step 5600) but has 2.3× fewer FLOPs per step
  (no energy-attention recurrence). It's the fastest arm at 1.2 s/step vs 3.2-6.6.
- **psd_anti > unconstrained** — E4 (3.83) vs E5 (3.95) at matched steps. Consistent ~0.07-0.10
  nats gap that stabilized. Confirms the other session's finding on megatron data.
- **E1 killed** — same model as E4, duplicate. 6.6 s/step = 13.5h was too slow for the deadline.
  E4 (1B probe) gives the dense loss curve at shorter budget.

---

## µP Probe Results (500 steps each, 2 GPUs)

Final `train-loss` at step 500. Architecture: 6G1x6E w1w2 sparse K=8 k=2.
Widths scaled proportionally: I_mlp ∝ d, I_e ∝ d, heads = d/64.

| LR | d=256 (35M) | d=384 (61M) | d=512 (90M) | d=768 (163M) |
|---|---|---|---|---|
| 1e-2 | 5.996 | 6.573 | 6.423 | (running) |
| 5e-3 | 5.966 | 5.872 | — | (running) |
| **2e-3** | **5.527** | 5.709 | — | (running) |
| **1e-3** | 5.649 | **5.532** | **5.437** | (running) |
| 5e-4 | 5.923 | 5.694 | — | (running) |

**Optimal LR per width:**
- d=256: **2e-3** (loss 5.527)
- d=384: **1e-3** (loss 5.532)
- d=512: **1e-3** (loss 5.437)
- d=768: pending

µP theory predicts optimal LR ∝ 1/d. d=256→384 shows a 2× drop (2e-3 → 1e-3).
d=384→512 shows NO drop (stays 1e-3). Need d=768 to distinguish:
- If d=768 optimal = 1e-3 → d=1024 should try 1e-3 and 5e-4
- If d=768 optimal = 5e-4 → classic µP holds, d=1024 ≈ 3e-4

12/14 probes complete. d=768 results expected ~02:45.

---

## Parameter accounting at d=768

| component | params | % of 162M |
|---|---|---|
| tied embedding (100352 × 768) | 77.07M | 47.6% |
| 6× softmax attention (4d²) | 14.16M | 8.7% |
| 6× dense swiglu MLP (3×d×2048) | 28.31M | 17.5% |
| 1× energy attention (2d²) | 1.18M | 0.7% |
| 1× w1w2 MoE K=8 I_e=3361 | 41.32M | 25.5% |

The MoE block is 25.5% of total — meaningful but not dominant. At d=1024/350M, embedding
drops to 29% and the MoE block grows to ~35%, giving the mechanism more room.

---

## Cluster status (02:40 UTC Sep 25)

- **grp_ebm (32 GPUs):** abl_AB (8, 94% done ~06:00) + E2/E3/D2/E4/E5 (20) + µP probes (4)
- **preemptable:** abl_Z1 (running)
- **abl_AB finishes ~06:00** → frees 8 GPUs for d=1024 runs

## Next steps

1. **~03:00:** E3 (Switch) finishes → submit eval, compare with E2/D2 at matched steps on wandb
2. **~03:30:** E4/E5 finish → psd_anti vs unconstrained verdict
3. **~06:00:** abl_AB finishes → 8 GPUs free for d=1024
4. **~07:30:** D2 finishes → iso-FLOP sparse result
5. **~09:30:** E2 finishes → iso-total sparse vs Switch comparison (the paper question)
6. **After E2:** submit d=1024 runs with µP-derived LR, all-MoE variants (12S, 6S1x6E, 6S6E)

## Fixes applied this session

1. `submit_train.sh`: NCCL_DEBUG_OVERRIDE unbound variable bug fixed
2. `compile_helpers()`: ninja skip when .so already loaded (was hanging 10-30 min on loaded FS)
3. `abl_X_16gpu`: load_lr_scheduler: false (root cause of 5 failed 16-GPU attempts)
4. CLAUDE.md: configs go in `configs/iclr_26/` until deadline
5. `fork-code-analyst` agent created at `~/.claude/agents/`
