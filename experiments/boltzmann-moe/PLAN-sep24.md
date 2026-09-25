# PLAN: FineWeb sweep — FINAL RESULTS (2026-09-25 22:30 UTC)

**Data:** nemotron-cc web-hq p2_0 (granite-4.0 tiktoken). NOT FineWeb-Edu (see DATA_NOTE.md).
**LR:** 1e-3 (µP-derived). **Schedule:** cosine, no constant phase. **GSM8K:** skipped.

## Final Results — all iso-active, all tokens matched within each scale

### d=768, ~132M active, 4.0B tokens

| arm | arch | active | FLOPwt | loss | Avg11 | MMLU | PPL |
|---|---|---|---|---|---|---|---|
| Switch MoE † | 6G1x6S | 132M | 196M | 3.216 | **44.72** | 24.1 | 42.1 |
| 12-layer dense † | 12G | 162M | 162M | 3.164 | 43.95 | 26.3 | 39.2 |
| Boltz-MoE rec | 6G1x6E | 132M | 194M | 3.283 | 43.96 | 25.1 | 45.4 |
| Boltz-MoE deep (iso-act) | 6S6E | 132M | 132M | 3.287 | 43.47 | 24.5 | 46.7 |

### d=1024, ~225M active, 7.6B tokens

| arm | arch | active | FLOPwt | loss | Avg11 | MMLU | PPL |
|---|---|---|---|---|---|---|---|
| Switch MoE † | 6G1x6S | 225M | 458M | 2.988 | **46.01** | 24.8 | 31.4 |
| 12-layer dense † | 12G | 255M | 255M | 3.057 | 45.56 | 25.6 | 34.4 |
| 12S all-Switch † | 12S | 225M | 225M | 3.048 | 45.51 | 25.6 | 33.3 |
| Boltz-MoE deep (iso-act) | 6S6E | 225M | 225M | 3.151 | 45.07 | 24.6 | 35.1 |
| Boltz-MoE rec | 6G1x6E | 225M | 454M | 3.101 | 44.73 | 22.9 | 36.2 |
| 12N EGPT (no MoE) | 12N | 318M | 318M | 3.288 | 42.77 | 23.8 | 47.5 |

### d=1280, ~315M active, 5.6B tokens

| arm | arch | active | FLOPwt | loss | Avg11 | MMLU | PPL |
|---|---|---|---|---|---|---|---|
| Switch MoE † | 6G1x6S | 315M | 661M | 2.954 | **46.89** | 24.6 | 30.0 |
| Boltz-MoE deep (iso-act) | 6S6E | 315M | 315M | 3.057 | 46.17 | 25.1 | 34.3 |
| 12-layer dense † | 12G | 363M | 363M | 3.016 | 46.16 | 25.7 | 32.9 |
| Boltz-MoE rec | 6G1x6E | 314M | 655M | 3.098 | 45.45 | 26.0 | 34.3 |

## Key findings

1. **Switch leads at every scale** (44.72 → 46.01 → 46.89) but the gap to 6S6E narrows:
   d=768: 1.25pp, d=1024: 0.44pp (vs 12S), d=1280: 0.72pp.

2. **6S6E deep matches 12G baseline at d=1280** (46.17 vs 46.16) with 13% fewer FLOPs
   (315M vs 363M) and no recurrence.

3. **6S6E deep beats Boltz-MoE recurrent** at d=1024 (45.07 vs 44.73) and d=1280 (46.17 vs
   45.45) while using 2× fewer FLOPs. Recurrence doesn't help over depth.

4. **12S all-Switch (45.51) ≈ 12-layer dense (45.56)** at d=1024 — the learned gate in a
   deep all-MoE stack matches a dense model at iso-active.

5. **6S6E vs 12S at iso-active/iso-FLOP/iso-token (d=1024):** 45.07 vs 45.51 = −0.44pp.
   Energy routing trails the learned gate by less than half a percentage point.

6. **Token efficiency:** Switch reaches Boltz's final loss at ~53% of Boltz's token budget.
   Boltz would need ~1.8× tokens to match 12G. EGPT (no MoE) is 4.2× less efficient.

7. **Boltzmann routing is free:** 75% of expert GEMMs are shared with routing (app:shared-compute).

## 32 GPUs now free on grp_ebm. Next steps if time permits:
- 1.5× Chinchilla at d=1280 (all arms, ~4h each at 16 GPUs)
- 12S baseline at d=1280
- Larger scale (d=1536)
