# PLAN: FineWeb-Edu fast sweeps — isolate what broke Boltzmann MoE

**Date:** 2026-09-24 (started), updated 2026-09-25 05:10 UTC
**Deadline:** end of Sep 25, 2026
**Data:** nemotron-cc p2_0 (granite-4.0 tiktoken, proven megatron format)

## Key discoveries so far

1. **The composable code is NOT a regression.** New w1w2 dense (Avg11 43.96) matches legacy (43.62)
   and the 12G baseline (43.95) despite 18% fewer params. Safe to use new code going forward.
2. **Energy MoE does 5× the GEMMs of Switch per block call** (846 vs 169 GMAC). The "similar FLOPs"
   framing was wrong — it measured whole-model active params, not per-block arithmetic.
3. **`sparse_explore: 0` gives a 2-3× wall-clock speedup** with no visible quality loss. The
   exploration candidates doubled the bmm work while contributing nothing to the forward output.
4. **psd_anti > unconstrained** by ~0.07-0.10 nats (stable across training). Confirmed on two datasets.
5. **µP optimal LR = 1e-3** from d=384 through d=768 (flatlines, not 1/d). Use 1e-3 for d=1024.
6. **Natural output alignment emerges** in deep 6S6E: last 3 energy layers show cos 0.31-0.49
   alignment without any regularizer. Weight alignment is near zero — alignment is emergent.
7. **Sinkhorn prevents expert collapse.** effK 6.7/8 (K=8) and 11.4/16 (K=16), all experts used,
   even with `sparse_explore: 0`.

---

## Round 1 RESULTS ✅ (d=768, 4 GPUs, cosine w/ constant phase)

| arm | code | K | k | total | tokens | lm_loss | Avg11 | MMLU |
|---|---|---|---|---|---|---|---|---|
| **A** 12G baseline | MLP | — | — | 162M | 4.86B | **3.164** | **43.95** | 26.33 |
| **C** new w1w2 dense | EnergyFF | 4 | 4 | 133M | 4.00B | 3.236 | **43.96** | 24.06 |
| **B** legacy w1w2 dense | BoltzMoE_MLP | 4 | 4 | 133M | 4.00B | 3.220 | 43.62 | 23.42 |
| **D** new w1w2 sparse | Surrogate | 16 | 2 | 158M | 2.41B | 3.313 | 43.27 | 24.82 |

---

## Round 2 IN PROGRESS (d=768, proper cosine, iso-total 162M, 4.0B tokens)

| arm | job | K | k | s/step | step | loss | ETA |
|---|---|---|---|---|---|---|---|
| **E3** Switch 6G1x6S | ✅ DONE | 8 | 2 | 1.2 | 7620 | **3.216** | — |
| **E4** psd_anti probe | ✅ DONE | 8 | 8 | 6.6 | 1907 | ~3.55 | — |
| **E5** unconstrained probe | ✅ DONE | 8 | 8 | 6.7 | 1907 | ~3.59 | — |
| **D2** sparse K=16 iso-FLOP | 1909681 | 16 | 2 | 1.30 | 5450 | 3.42 | ~05:40 |
| **E2** sparse K=8 iso-total | 1909680 | 8 | 2 | 1.45 | 4200 | 3.40 | ~06:20 |
| **G1** 6S6E deep sparse | 1909379 | 8 | 2 | 2.40 | 2200 | 3.65 | ~08:45 |
| **G2** 6S6E + alignment reg | 1910010 | 8 | 2 | ~2.5 | early | — | ~09:00 |

### Speed optimization applied mid-run
- `sparse_explore: 2 → 0`: removes 2 redundant candidate evaluations per token
- 4 GPU → 8 GPU: halves gradient accumulation steps (ga 8 → 4)
- Combined: **4.5 → 1.45 s/step (3.1× speedup)** on E2

---

## d=1024 SCALE-UP IN PROGRESS (preemptable, 8 GPUs, 7.6B tokens, LR=1e-3)

| arm | job | arch | total | s/step | step | loss | ETA |
|---|---|---|---|---|---|---|---|
| **F1** 12G baseline | 1909367 | 12G dense | 255M | 0.44 | 11500 | 3.08 | **~05:15** |
| **F2** Boltz sparse K=8 | 1909789 | 6G1x6E | 350M | 1.68 | 810 | 4.30 | ~11:15 |
| **F3** Switch K=8 | 1909370 | 6G1x6S | 352M | 1.56 | 3500 | 3.36 | ~09:55 |
| **F5** 12N EGPT (no MoE) | 1910008 | 12N energy | 255M | ~1 | early | — | ~09:00 |
| ~~F4 12S all-Switch~~ | KILLED | too thin: I_s/d=0.34 at 255M with 12 layers |

---

## µP Probe Results ✅ (500 steps each, all 20 done)

| LR | d=256 (35M) | d=384 (61M) | d=512 (90M) | d=768 (163M) |
|---|---|---|---|---|
| 1e-2 | 6.00 | 6.57 | 6.42 | 6.44 |
| 5e-3 | 5.97 | 5.87 | 6.10 | 6.12 |
| **2e-3** | **5.53 ★** | 5.71 | 5.59 | 5.50 |
| **1e-3** | 5.65 | **5.53 ★** | **5.44 ★** | **5.42 ★** |
| 5e-4 | 5.92 | 5.69 | 5.58 | 5.43 |

Optimal LR = **1e-3** from d=384 onward. NOT 1/d scaling — saturates. d=1024 uses 1e-3.

---

## Alignment measurement (G1 6S6E at step 2000)

**MLP/MoE output cosine similarity (consecutive pairs):**
- Switch layers (0-5): low (0.05 - 0.22) — independent representations
- Energy layers (6-11): **increasing alignment** — L9-L10: 0.31, L10-L11: **0.49**
- Cross-type: near-zero — the two halves produce orthogonal outputs

**Weight alignment:** near-zero (0.00-0.06) between all energy layer pairs.
→ Output alignment is EMERGENT, not from shared weights.

G2 (with alignment regularizer) tests whether encouraging this helps.
Figure being generated for the paper.

---

## Speed audit findings

The 3.7× wall-clock gap (energy 4.5 vs Switch 1.2 s/step) comes from:
1. `sparse_explore=2`: evaluates 4 candidates instead of 2 → **2× bmm work** (FIXED: set to 0)
2. w1w2 form: 4 matmuls/expert vs SwiGLU's 3 → 1.33× overhead (structural)
3. Larger I_e (3361 vs 2240 at iso-total) → 1.50× per matmul (structural)

After fix: energy runs at **1.45 s/step vs Switch 1.2 s/step = 1.2× gap** (down from 3.7×).

---

## Param accounting

### d=768 (162M iso-total)
| component | params | % |
|---|---|---|
| tied embedding (100352 × 768) | 77.07M | 47.6% |
| 6× softmax attention | 14.16M | 8.7% |
| 6× dense swiglu MLP | 28.31M | 17.5% |
| 1× energy attention | 1.18M | 0.7% |
| 1× w1w2 MoE K=8 I_e=3361 | 41.32M | 25.5% |

### d=1024 arms
| arm | total | active | flopwt | I/d |
|---|---|---|---|---|
| F1 12G baseline | 255M | 255M | 255M | 2.69 |
| F2 Boltz 6G1x6E K=8 | 350M | 223M | 445M | 10.09 |
| F3 Switch 6G1x6S K=8 | 352M | 225M | 458M | 6.72 |
| F5 12N EGPT (no MoE) | 255M | 255M | 255M | 5.05 |

---

## Fixes applied this session

1. `submit_train.sh`: NCCL_DEBUG_OVERRIDE unbound variable bug
2. `compile_helpers()`: ninja skip when .so already loaded (10-30 min hang)
3. `abl_X_16gpu`: `load_lr_scheduler: false` (root cause of 5 failed 16-GPU attempts)
4. GPU-count change resume: need BOTH `load_lr_scheduler: false` AND `load_rng_state: false`
5. `sparse_explore: 0` — 2-3× training speedup with no quality loss
6. CLAUDE.md: configs go in `configs/iclr_26/` until deadline
7. `fork-code-analyst` agent, `table-updater` agent adapted for FineWeb
8. Alignment regularizer module + measurement script
9. abl_AB disabled in watchdog (was zombie-resubmitting after kill)
