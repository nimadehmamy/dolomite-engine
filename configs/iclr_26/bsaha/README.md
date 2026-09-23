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
