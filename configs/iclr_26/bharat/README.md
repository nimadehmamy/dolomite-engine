# For Bharat — a 1B pair where EVERY layer is a routed MoE

**★ MAIN SUGGESTION.** Two configs, an experiment **only as a pair**: identical datamix, budget,
tokens/step, schedule, width, depth and attention. The **only** difference is the FFN router.

| config | GPUs | total | ACTIVE | MoE layers | K | k | I_e | k·I_e / d |
|---|---|---|---|---|---|---|---|---|
| `bharat_1B_18L_allMoE_boltz.yml` | 8 | **1022.4M** | **512.8M** | **18 of 18** | 32 | 8 | 768 | **4.0×** |
| `bharat_1B_18L_allMoE_switch_baseline.yml` | 8 | **1004.4M** | **494.8M** | 18 of 18 | 32 | 8 | 256 | — |

d=1536, 18 layers, 24 heads (head_dim 64 = rope_dim 64), **softmax attention in every block**.
524,288 tok/step × 61,035 = **32.0B**. Both meta-build and agree with `audit_config` to the byte.
Switch uses `I_s = I_e/3 = 256` because swiglu stores three matrices per expert and hopfield one —
that is what makes them iso-parameter at the same K and k.

## Why this shape, and why not ours

Built to match what MoE models actually do, verified from their own `config.json`:

| model | L | d | MoE layers | K | k | I_e | k·I_e / d |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 16 | 2048 | **all 16** | 64 | 8 | 1024 | **4.0×** |
| Qwen1.5-MoE-A2.7B | 24 | 2048 | **all 24** (+shared) | 60 | 4 | 1408 | 2.8× |
| our 400M hybrid | 7 | 1024 | 1 of 7 (×6 rec) | 32 | 2 | 5871 | **11.5×** |
| our existing 1B | 12 | 1024 | 4 of 12 | 64 | 2 | 2846 | 5.6× |
| `fallback/` 8S4E | 12 | 1024 | 12 of 12 | 32 | 8 | 3000 | 23.4× |

The field puts an MoE in **every** layer and keeps the effective per-token FFN width `k·I_e` near
**4×d** — the width a dense model's FFN would have. Our arms invert that: very few MoE layers, each
absurdly wide. This config follows the field.

Note this **retires an old worry of ours** that `I_e=1024` is "too narrow for sparsity to pay":
OLMoE uses exactly `I_e=1024` and works, because `k=8` and `K=64` make `k·I_e` right. Narrow experts
are fine; `k·I_e` is what matters.

## The number that motivates all of this

**Our existing 1B activates like a 400M model, which is why it performs like one.**

| | total | ACTIVE | bank | bank ACTIVE | Avg11 | ppl |
|---|---|---|---|---|---|---|
| `cmix1B_12L_gptDense_32B` (K=64, k=2) | 1002.1M | 279.3M | 746.1M | **23.3M (3.1%)** | 47.79 | 29.46 |
| `abl_B_400M_6G1x6S` (our best arm) | 400.0M | 219.7M | 192.4M | 12.0M | **48.24** | **28.27** |
| **`bharat_1B_18L_allMoE_boltz`** | 1022.4M | **512.8M** | 442.4M | **110.6M (25%)** | — | — |

The old 1B stores 746M of experts and uses 23M per token. Its ACTIVE count is only **+27%** over the
400M arm and it **loses** to that arm on every column.

## Choices you may want to overrule

- **K=32, k=8.** K=8 (the earlier suggestion) is too few experts to route among at this width;
  K=64/k=2 is the trap above. `p = k+4 = 12` of 32, so the sparsity ceiling is `K/p = 2.7×`.
- **d=1536, not 2048.** More standard would be 2048, but our vocab is **100,352** — 3× TinyLlama's —
  so the tied embedding alone costs 205.5M at d=2048 and total passes 1.5B. At d=1536 it is 154.1M.
- **COSINE, not WSD.** No controlled WSD-vs-cosine A/B exists at 32B; WSD was confirmed only on short
  probes. What matters is that the pair shares one schedule, which it does (`2000 / 0 / 59035`).
- **hopfield + rank-16 subspace PROXY, not the surrogate head.** Measured at 134M, same budget:
  proxy **0.20 s/step** vs surrogate **0.30** — the proxy is **1.5× faster**, and the quality gap
  (Avg11 45.01 vs 44.82) is inside noise.

## For 128B tokens

`num_training_steps` → **244,140** and `num_decay_steps` → **242,140**, so
`warmup + constant + decay == num_training_steps`. Not optional: published `slope90k_*` configs once
ran 60,000 of 90,000 steps pinned at the LR floor because that was violated.

## Launch

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
export WANDB__SERVICE_WAIT=300
for a in bharat_1B_18L_allMoE_boltz bharat_1B_18L_allMoE_switch_baseline; do
  bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \
       $a configs/iclr_26/bharat/$a.yml 8 preemptable 24:00 200G 4
done
```

Verify within 60 s — **step lines are in STDERR**, and `grep -m1 DeviceMesh <stderr>` must NOT say
`'cpu'` (a host with failed CUDA init makes the trainer fall back to a CPU mesh and train nothing
while LSF reports RUN). Sub-60-second multi-node deaths are transient (~50%): just resubmit.

## `fallback/` — the earlier 8S4E pair

`bharat_1B_8S4E_boltz` (998.7M / 364.0M active) and `bharat_1B_12S_switch_baseline` (998.1M /
364.7M). Kept because they are verified and matched, but **superseded**: 8S4E puts `k·I_e` at 23×d,
nothing like field practice, and reaches only 364M active against this pair's 513M at the same total.
Use them only if the main pair hits a problem.

---

# 400M deep all-MoE (6S6E), 128B tokens, 4 nodes — added 2026-09-23

`bharat_400M_6S6E_deep_allmoe_128B_4node.yml`

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh bharat_6S6E_128B \
     configs/iclr_26/bharat/bharat_400M_6S6E_deep_allmoe_128B_4node.yml 32 preemptable 24:00 400G
```

Pass `32` and let the submitter turn it into `-n 4 -gpu num=8/task -R span[ptile=1]` + `blaunch`.
`-gpu "num=32/task" -n 1` asks for 32 GPUs on ONE host and PENDs forever.

## Architecture

12 distinct layers, **no recurrence**. 6 standard sparse-MoE blocks (K=8, top-4,
`intermediate_size: 1024` **per expert**, 1.00x hidden), then 6 distinct energy Boltzmann-MoE blocks
(K=32, top-2, `intermediate_size: 187872` **total across experts**, I_e=878). Note that asymmetry in
what `intermediate_size` means; it has cost a run here.

## Budget

`122070 steps x 1,048,576 tok/step = 128.0B`, where `tokens/step = 32 x mbs 4 x ga 2 x 4096`.
**GPUS is not in the file** — at 8 GPUs this gives a quarter of the budget. `tokens/call = 16,384`,
the column where our `sparse_forward` wins 2.1-2.4x (below ~16,384 it loses), so do not lower `mbs`
without raising `sequence_length`.

## Parameters (millions, `audit_config`)

Matched to our dense-trunk 6G6E arm on ACTIVE and FLOPwt, deliberately **not** on TOTAL:

| arm | TOTAL | ACTIVE | FLOPwt |
|---|---|---|---|
| 6G6E dense trunk | 399.85 | 238.02 | 238.02 |
| **this arm (6S6E)** | **475.40** | **238.07** | **238.07** |

No sizing can match all three. A dense block has `active == total`; a top-k MoE block has
`active == (k/K)*total`, so converting the trunk at fixed total must lose active. Swept
K_S x k_S x I_S: every iso-total sizing loses >=15.8% ACTIVE, every iso-active sizing costs >=1.19x
TOTAL, no overlap. Neither arm uses recurrence, so `FLOPwt == ACTIVE` identically.

## Multi-node: safe, with the evidence

- `repulsion_space: output` + `repulsion_subsample: 64` is the multi-node-safe repulsion. **Do not
  switch to `weight`** — it wedges on >1 node silently (zero NCCL errors, zero steps), because
  regrouping a dim-0-sharded weight by expert crosses shard boundaries and issues an inter-node
  collective from inside a dynamo graph.
- `fused_experts: true` **is** fine multi-node. It was once recorded as single-node-only; that was
  wrong and the correction is measured (fused + sparse + output-space repulsion, 120-step 2-node
  probe, 1.44 s/step). It also cannot be turned off: `sparse_forward` and `proxy_rank > 0` both
  assert on it.
- `stage: 0` — no FSDP sharding, pure DDP replication.
- The submitter exports `NCCL_IB_DISABLE=1` for multi-node, because this cluster's InfiniBand fails
  the first inter-node RDMA (`IBV_WC_RETRY_EXC_ERR`, 6+ HCAs, 7+ peers, 4 host pairs) and
  `NCCL_IB_TIMEOUT`/`RETRY_CNT` do not fix it. Inter-node traffic goes over TCP. Arms launched
  without it sat dead at step 0.
- **~50% of multi-node launches die at STARTUP** with a LOUD `ncclRemoteError` at the first
  collective, before training. Just resubmit. Do not confuse it with the silent weight-space wedge.

## Speed, and the one lever

Our 6G6E arm at the same FLOPwt runs 1.736 s/step at 8 GPUs with 65,536 tokens/rank. This file puts
32,768 tokens/rank, so compute should be ~0.9 s/step, plus one full-gradient all-reduce per step over
TCP (stage 0 replicates, so the whole ~475M gradient is reduced). **Expect ~1.2-1.7 s/step, i.e.
41-58 h for 128B.** If comms dominate, raise `ga` to amortise the all-reduce: `mbs 4 / ga 4` at 32
GPUs gives 2,097,152 tok/step, so set `num_training_steps: 61035` and `num_decay_steps: 59035` to
keep 128B. Do **not** trust a single multi-node s/step reading — the same config has measured
2.45-6.81 s/step across placements, a 2.7x spread from placement alone.

## Data

`/proj/dmfexp/datasets-shared/granite-4-cmix-FULL`, 570.7B tokens, validated 2026-09-23, group
`proj_dmfexp`. **On launch the trainer logs `Tokens per epoch:` per shard — the two web shards must
read ~265-266e9.** ~50e9 means the biased 18.6%-prefix subset, ~0.47 nats easier and not comparable.

128B is 0.22 epochs of the blend overall, but `megamath-web-pro_0` has only 13.0B tokens in existence
and receives 0.15 x 128B = 19.2B, so it is read **~1.48 times**. Known and accepted for this mixture
at 128B, but it does mean "no arm revisits a document" is false for this run.

## Still on the doomed data path

The 1B configs here (`bharat_1B_18L_allMoE_{boltz,switch_baseline}.yml` and both `fallback/` files)
still name `/proj/datasets/granite-4-datasets-megatron-merged`. They were left alone deliberately in
case a run is live on them: repointing changes the Megatron blend index, which is keyed on the paths,
so a restart would rebuild it and change the sample order. **Repoint them to
`/proj/dmfexp/datasets-shared/granite-4-cmix-FULL` before the next launch** — the owners may delete
that tree at any time, and a job that is running survives it only until its next requeue.
