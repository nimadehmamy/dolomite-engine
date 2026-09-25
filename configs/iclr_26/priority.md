# Experiment priorities — FineWeb-Edu sweep (2026-09-25)

## Priority rules
- **Deadline: end of Sep 25, 2026**
- Total params under 700M; train 1-1.5× Chinchilla in <18h
- Three active-param scales: ~130M (d=768), ~250M (d=1024), ~400M (d=1280)
- Compare: Boltzmann 6G1x6E vs Switch 6G1x6S vs 6S6E deep (no recurrence)
- 12G baseline ONLY at d=768 (the iso-active question is settled there)
- `psd_anti` for all energy models (confirmed > unconstrained by ~0.07 nats)
- `sparse_explore: 0` for all sparse arms (confirmed: 2-3× speedup, no quality loss)
- LR=1e-3 for all scales (µP probes: optimal from d=384 to d=768, use for d=1024/1280)

---

# FineWeb sweep

## P0 — RUNNING, highest priority (our GPUs)

### Scale 1: d=768, 4.0B tokens, grp_ebm
| arm | arch | total | active | status | queue |
|---|---|---|---|---|---|
| E3 Switch 6G1x6S | 6G1x6S | 163M | 132M | ✅ DONE Avg11=44.72 | — |
| E2 Boltz 6G1x6E K=8 | 6G1x6E | 162M | 141M | ✅ DONE, eval pending | — |
| D2 Boltz 6G1x6E K=16 | 6G1x6E | 168M | 127M | ✅ DONE lm_loss=3.322 | — |
| G1 6S6E deep | 6S6E | 162M | 122M | ▶ 70% | grp_ebm 4GPU |
| G2 6S6E+align | 6S6E | 162M | 122M | ▶ 40% | grp_ebm 4GPU |

### Scale 2: d=1024, 7.6B tokens
| arm | arch | total | active | status | queue |
|---|---|---|---|---|---|
| F1 12G baseline | 12G | 255M | 255M | ✅ DONE Avg11=45.56 | — |
| F3 Switch 6G1x6S | 6G1x6S | 352M | 225M | ▶ 50% | preemptable 8GPU |
| F2 Boltz 6G1x6E | 6G1x6E | 350M | 265M | ▶ 25% | preemptable 8GPU |
| F5 12N EGPT | 12N | 318M | 318M | ▶ 40% | preemptable 8GPU |

### Scale 3: d=1280, 11.2B tokens
| arm | arch | total | active | status | queue |
|---|---|---|---|---|---|
| H1 12G baseline | 12G | 363M | 363M | ▶ 8% | grp_ebm 8GPU |
| H2 Boltz 6G1x6E | 6G1x6E | 500M | 375M | ▶ 3% | grp_ebm 8GPU |

## P1 — NEXT TO SUBMIT (our GPUs, as they free up)

| arm | arch | d | why | queue |
|---|---|---|---|---|
| H3 Switch 6G1x6S d=1280 | 6G1x6S | 1280 | Switch baseline for Scale 3 | grp_ebm or preemptable 8GPU |
| 6S6E deep d=1024 | 6S6E | 1024 | Deep allMoE at Scale 2 | preemptable 8GPU |

## P2 — FOR BSAHA (preemptable, lower priority)

Configs in `configs/iclr_26/bsaha/fineweb/`.
| arm | arch | d | total | notes |
|---|---|---|---|---|
| 12S all-Switch d=768 | 12S | 768 | 162M | All-Switch baseline for 6S6E comparison |
| 6S6E+align d=1024 | 6S6E | 1024 | ~350M | Alignment regularizer at larger scale |
| 12N EGPT d=768 | 12N | 768 | 162M | Pure energy model baseline at d=768 |

## P3 — FOR BHARAT (needs >16 GPUs)

Configs in `configs/iclr_26/bharat/fineweb/`.
| arm | arch | d | total | notes |
|---|---|---|---|---|
| 6S6E deep d=1536 | 6S6E | 1536 | ~700M | Largest deep allMoE, needs 16-32 GPUs |
| 12S baseline d=1536 | 12S | 1536 | ~500M | All-Switch baseline at largest scale |

---

# Megatron (old experiments, lower priority)

The cmix/megatron runs from before Sep 24 are documented in HANDOFF.md §1-28.
Those use the 70/30 web/math datamix and 32B token budget.
The FineWeb runs above supersede them for the scaling comparison.

Key results to preserve from the Megatron era:
- cmix_134M_hybrid: Avg11=44.82 (the published number)
- abl_B Switch 6G1x6S: Avg11=45.43 (FLOP-matched, Switch leads by 0.61pp)
- abl_E base EGPT: Avg11=45.87 (no MoE, best at 134M)
