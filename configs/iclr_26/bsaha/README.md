# 400M configs for bsaha

Updated 2026-09-23. **Every config here reads
`/proj/dmfexp/datasets-shared/granite-4-cmix-FULL`**, an independent full-corpus copy (570.7B
tokens, validated 2026-09-23, mode 444, readable by POSIX group `proj_dmfexp`). The originals named
`/proj/datasets/granite-4-datasets-megatron-merged`, which the owners may delete at any time.

## Launch

Never hand-roll the bsub. `-gpu "num=16/task" -n 1` asks for 16 GPUs on ONE host and PENDs forever;
hosts here have 8. The submitter encodes the node shape, the ptile/blaunch rules and the resume:

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh <jobname> <config> 8 preemptable 24:00 400G
```

**`tokens/step = GPUS x micro_batch_size x gradient_accumulation_steps x sequence_length`, and GPUS
IS NOT IN THE CONFIG.** Every 400M file here wants **exactly 8 GPUs** for its 32.0B budget; at 4 it
silently trains on half and is not comparable to anything in our tables.

**Sanity check on launch:** the trainer logs `Tokens per epoch:` per shard. The two web shards must
read ~265-266e9. ~50e9 means the biased 18.6%-prefix subset, whose loss is ~0.47 nats easier.

## What to run first

| config | why | TOTAL / ACTIVE / FLOPwt (M) |
|---|---|---|
| **`bsaha_400M_hybrid_6G1x6E_rnormnone_s7.yml`** | **seed replicate of our live 400M hybrid.** We have 2 seeds at 134M and only 1 at 400M, and the 134M pair showed 0.16pp Avg11 spread on the oracle path but **0.87pp on the deployed sparse path** | 399.8 / 219.4 / 299.4 |
| **`bsaha_400M_sandwich_isoall.yml`** | **the sandwich matched on all three axes.** Without it the 400M sandwich row stays 28% short on compute | 399.5 / 219.2 / 299.1 |

Parameters in millions; for the pair above the point is EQUALITY, not a maximum.

## Retiring arms, kept for reference

| config | status |
|---|---|
| `abl_Y_400M_sandwich_rnorm_none.yml` | iso-TOTAL only: `layer_iterations [1,4,1]` = 6 block applications vs the hybrid's 12, so **ACTIVE -28.7%, FLOPwt -27.5%**. Any "sandwich loses to hybrid" read from it is confounded with a 28% compute deficit. Superseded by `bsaha_400M_sandwich_isoall.yml` |
| `abl_H_400M_6G6E_deep.yml` | deep energy arm on `routing_norm: zscore`; superseded by our rnorm=none rerun. Keep as the zscore reference for that ablation |
| `abl_H_400M_6G6S_deep.yml` | deep Switch baseline. **Weak baseline:** its MoE trunk is `intermediate_size: 290` per expert = 0.28x hidden, inside the narrow-expert band that has repeatedly failed here |
| `abl_H_400M_6G6G_deep_isoactive.yml` | deep dense control, sized iso-ACTIVE with the energy arm |
| `abl_G_400M_*`, `cmix_400M_*` | older arms, data path updated only |
| `abl_C_134M_1G1x6E1G_isototal.yml` | **134M, not 400M, and it wants 4 GPUs, not 8** (mbs 4 / ga 4 -> 262,144 tok/step at 4 GPUs). At 8 it gives 64B, double budget |

## Settings worth knowing before you change anything

- **`routing_norm: none`** is our largest knob: at 134M +1.49pp Avg11 on the block-position variant,
  +0.10pp on the hybrid, it collapses a 1.37pp block-placement penalty to 0.02pp, does NOT collapse
  routing (least-used expert 0.26% -> 5.1%, effK 13.79 -> 15.83 of 16), and is ~24% faster. It IS
  worse on train loss (+0.047 nats) — for this knob loss has been the wrong signal three times, so
  **select on Avg11, not loss**.
- **`mbs 4 / ga 4`, not `2 / 8`.** Same tokens/step, but tokens/call goes to 16,384 where
  `sparse_forward` wins 2.1-2.4x; below ~16,384 it LOSES (0.41-0.49x). Measured 6.397 -> 3.783 s/step.
- **`repulsion_space: output` + `repulsion_subsample: 64`** if you go multi-node.
  `repulsion_space: weight` wedges silently on >1 node (zero NCCL errors, zero steps).
- **`fused_experts: true` must stay on.** It is not an optional speed knob: `sparse_forward` and
  `proxy_rank > 0` both assert on it, and with it off the proxy router silently never trains.
- **`stage: 0`** — no FSDP sharding, pure DDP replication, correct at this scale.
- **`intermediate_size` is TOTAL across experts for the energy blocks but PER-EXPERT for
  `mlp_type: MoE`.** That asymmetry has cost a run here.

---

## Multi-node transport: native InfiniBand is the DEFAULT (2026-09-23)

`submit_train.sh` enables native IB automatically whenever `nnodes > 1`. **You do not need to pass
anything.** Measured placement-controlled, both transports on the same host pair, same config, 400
steps, zero genuine NCCL errors either side:

| regime | TCP | native IB | speedup |
|---|---|---|---|
| dense | 1.2861 s/step | 0.9160 s/step | 1.40x |
| **sparse (step 300+)** | 1.1448 s/step | 0.7069 s/step | **1.62x** |

Quote the sparse row: `sparse_start_step` is 300 out of tens of thousands of steps, so a real run is
sparse for >99% of its life.

### Falling back to TCP

```bash
FORCE_TCP=1 bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh <name> <cfg> <gpus> ...
```

(`ALLOW_IB=0` is a synonym.) Expect ~1.4-1.6x slower.

**Use it when** a job dies at the FIRST collective with a loud `ncclRemoteError` /
`IBV_WC_RETRY_EXC_ERR` **on repeated resubmits**. Historically ~50% of multi-node launches died that
way at startup and then ran fine next attempt, so **resubmit once first** — the watchdog resubmits by
itself anyway. Only make TCP persistent if the same job keeps failing.

**Do NOT use it** for a *silent* stall at step 0 with zero NCCL errors. That is a different bug, the
weight-space repulsion wedge, and TCP will not fix it. Keep `repulsion_space: output` (never `weight`)
on anything multi-node.

**Checking for a real error is not a plain grep.** Our own scripts’ comments contain the strings
`ncclRemoteError` and `IBV_WC_RETRY_EXC_ERR`, and the generated job script is echoed into the log, so
a bare grep reports phantom failures — this misled me twice. Require a genuine NCCL line, and check
**both** streams, since NCCL writes its banner and errors to stdout while step lines go to stderr:

```bash
grep -hE "IBV_WC_RETRY_EXC_ERR|ncclRemoteError" <job>.stderr <job>.stdout \
  | grep -E "[a-z0-9-]+:[0-9]+:[0-9]+|NCCL WARN"
```

Or just run `bash experiments/boltzmann-moe/scripts/ib_probe_report.sh <jobid>:<name>`, which encodes
this plus the dense/sparse split.

---

# RUN ORDER, with measured time estimates (2026-09-23)

Table 1 was regrouped by recurrence on 2026-09-23. Three of the four groups already have a
FLOP-matched Switch baseline, already evaluated, so **only one baseline was actually missing**.

Estimates are `61035 steps x measured s/step` at **8 GPUs**, taken from real logs (median over all
post-warmup steps, n in brackets) rather than modelled. Add wall-clock for preemption: the watchdog
resubmits and resumes from the last checkpoint, so a preemption costs only the steps since the last
save, but it does stretch the calendar time.

| # | config | why | s/step (basis) | est. 8-GPU time |
|---|---|---|---|---|
| **1** | `bsaha_1B_8G4S_switch_baseline.yml` | **NEW. The only missing baseline.** The 1B row stands alone with nothing to compare against. Matched to `cmix1B_12L_gptDense_32B` on ACTIVE and FLOPwt to 0.00% (279.31 vs 279.32) | ~1.78 (from `cmix1B` 1.699 measured, n=99, x1.05 for the Switch/energy ratio seen at 400M: 1.470 vs 1.395) | **~30 h** |
| **2** | `bsaha_400M_sandwich_isoall.yml` | De-confounds the 400M sandwich row. The current row is ACTIVE -29% / FLOPwt -27% vs the hybrid, so it mixes architecture with a 28% compute deficit | 3.0-3.4 (same energy block as `abl_X` applied 6x; `abl_X` measures 3.423, n=1386, and this has 8 apps not 12) | **51-58 h** |
| **3** | `bsaha_400M_hybrid_6G1x6E_rnormnone_s7.yml` | Seed replicate. Every 400M row is single-seed and the 134M deployed-path seed spread is **0.87pp**, so any 400M claim under ~0.9pp is inside noise | 3.423 (measured on `abl_X`, n=1386 -- byte-identical except the seed) | **~58 h** |

**Scheduling reality, stated plainly.** Only #1 finishes comfortably. #2 and #3 at ~58 h from a
2026-09-23 21:00 UTC start land around 2026-09-26 07:00 with roughly 10 h of margin against a
2026-09-26 17:00 deadline -- and that assumes no preemption. If only one 400M run can be done, **#2
beats #3**: a confounded row is a reviewer's opening, whereas a missing error bar is a stated
limitation.

**All three are 8 GPUs exactly.** `tokens/step = GPUS x mbs x ga x seq` and GPUS is not in the file,
so 4 GPUs silently halves the budget and makes the run incomparable.

## Other measured s/step at 8 GPUs, for reference

| arm | FLOPwt | apps | mbs/ga | s/step (n) |
|---|---|---|---|---|
| `abl_X_400M_hyb_rnorm_none` | 299.38 | 12 | 4/4 | 3.423 (1386) |
| `abl_B_400M_6G1x6S` (Switch, FLOP-matched) | 300.95 | 12 | 4/4 | 2.451 (2247) |
| `abl_Y_400M_sandwich` (off-budget) | 217.20 | 6 | 4/4 | 1.526 (2348) |
| `abl_H_400M_6G6S_deep` (Switch, deep) | 239.50 | 12 | 2/8 | 1.470 (4579) |
| `abl_AB_400M_6G6E_deep_rnormnone` | 238.02 | 12 | 2/8 | 1.395 (463) |
| `cmix1B_12L_gptDense_32B` | 279.32 | 12 | 2/8 | 1.699 (99) |

Note FLOPwt alone does not predict s/step: `abl_B_400M_6G1x6S` and `abl_X` are within 0.5% on FLOPwt
yet differ 1.40x in wall-clock, because the energy path carries the proxy router, Sinkhorn and
repulsion on top of the same nominal FLOPs.
