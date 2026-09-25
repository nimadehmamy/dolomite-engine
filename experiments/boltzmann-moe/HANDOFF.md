# Boltzmann-MoE — HANDOFF

**Entry point for a fresh session.** Read this first, then jump into the specific
doc you need. This file carries (a) orientation, (b) the things that exist
nowhere else, and (c) the 2026-09-12 session findings, which materially change
the project's conclusions.

Last updated: 2026-09-16 (true sparsity §12.9-12.12; eval-metric banner added).

> ## 📏 EVAL METRIC — THE REFERENCE IMPLEMENTATION IS A SCRIPT, NOT A DESCRIPTION
>
> **Every headline number in this project and in the ICLR paper is `Avg11`, and the reference is
> `experiments/eval_scripts/compute_avg11.py`.** Run that; do not re-derive the recipe by hand and
> do not read an "Avg" from an older doc without checking which convention it uses.
>
> ```bash
> python experiments/eval_scripts/compute_avg11.py <run_dir_or_unsharded_dir>
> ```
>
> * **11-task unweighted mean.** `acc_norm`: arc_challenge, arc_easy, hellaswag, openbookqa, piqa,
>   sciq. `acc`: boolq, copa, winogrande, race, lambada_openai.
> * **MMLU (acc) and GSM8K-CoT (flexible-extract) are reported SEPARATELY and are NEVER folded into
>   the mean.** WikiText likewise separate — and report it as **bits/byte**, not `word_perplexity`
>   (they differ by `exp(bpb * 3.7066)`, which turns a 1.55x regression into a 7.8x one).
> * **This is the colleagues' recipe**, so our numbers are directly comparable to the EGPT-RL / FET
>   series. Source of truth: `~/Code/GPT-experiments/projects/EGPT-RL/RESULTS.md:247-249`.
>   `compute_avg11.py` is validated against their stored Avg11 on the two shared
>   `math_egptdual` seed checkpoints to within 0.004pp.
> * The script is now TRACKED in the repo (2026-09-16). It reproduces the published values exactly
>   — independently re-evaluated this session: `pure_hop_T12_sink` **40.96** (§12.2) and
>   `iclr_hop_K32_top2_sink` **44.58** (§12.3).
> * **Two superseded conventions appear in older sections of this file — never mix them with
>   Avg11.** `avg9` (9 tasks, MMLU excluded, race/lambada absent — the pre-`pyarrow>=20` bug) and
>   `avg10` (10 tasks, MMLU INCLUDED, race/lambada EXCLUDED). Avg11 runs ~3pp BELOW avg10 because
>   race (~0.28) and lambada (~0.23) sit near chance at our scale: a scoring-convention gap, not a
>   model effect. Full detail in `CLAUDE.md`'s metric block and in the script's own docstring.

> ## 📄 ACTIVE PAPER — the ICLR draft is the ONLY paper we are writing right now
>
> **Path: `~/Code/overleaf/boltzmann-moe-ICLR-2026/`**
> (Overleaf git-bridge remote, branch `main`; the URL is in the clone's `git remote -v`.)
> Files: `main.tex` + `sec/{intro,theory,experiments,appendix}.tex`.
> Table locations: `tab:frontier`, `tab:pure`, `tab:cost`, `tab:threeway` in
> `sec/experiments.tex`; `app:frontier`, `app:eval` (metric definition), `app:scale400`,
> `app:threeway`, `tab:attribution` in `sec/appendix.tex`.
>
> **ALL of its numbers are `Avg11`** as of 2026-09-15 (migrated from `avg10` that day; see
> `AVG11_ICLR_MIGRATION.md`). MMLU and GSM8K-CoT are reported as SEPARATE columns, never in
> the mean.
>
> **⚠ Do NOT read the older drafts unless a task specifically requires them — they burn
> context and they use SUPERSEDED metric conventions.** The NeurIPS draft
> (`~/Code/energy/energy-GPT-neurips2026/`) and the talk
> (`~/Code/overleaf/energy-GPT-reformulation-2026/`) are **reference/archive only**; their
> tables are still on `avg10`/`avg9`. If you find yourself opening `paper_v2.tex` or
> `boltz_moe.tex` to answer an ICLR question, stop — you are in the wrong paper.

---

## 0. Doc map — read in this order

| Doc | What it is | When to read |
|---|---|---|
| **HANDOFF.md** (this) | Orientation + 2026-09-12 findings | always, first |
| **`ROUTING_SIGN_BUG_20260915.md`** | 🔴 The composable router is SIGN-INVERTED (selects worst-matching experts). Affects the whole ICLR grid + live 400M. Includes the chemical-potential reframing of the balance property | **before any routing claim** |
| **`HANDOFF.md` §11** | 🔴 The 2026-09-15 session: sign inversion, chemical-potential balancing, Sinkhorn, energy stability | **always, with §0** |
| **`PLACEMENT_ARTIFACT_20260915.md`** | 🔴 The "5x slower" figure is mostly host placement (6.76 → 2.49 s/step, same code). Invalidates the launch-overhead reading | **before any s/step claim** |
| `ACCEL_FINDINGS_20260915.md` | fused-GEMM 1.61x (single-node ONLY — wedges at 2 nodes), repulsion sweep, proxy router 0.942, ReLU negative result | before optimisation work |
| **`~/Code/overleaf/boltzmann-moe-ICLR-2026/`** | **★ THE ACTIVE PAPER.** `main.tex` + `sec/*.tex`; all numbers `Avg11` | whenever writing or quoting results |
| `AVG11_ICLR_MIGRATION.md` | What changed in the 2026-09-15 avg10→Avg11 migration, number by number | before touching any paper number |
| `CLAUDE.md` | Operational guide: config fields, how to submit runs, routing metrics, results summary | before touching configs or launching anything. **Its "Key source files" table is stale — see §10** |
| `PROGRESS.md` | Full experiment log B/C/H series + the master baseline table (lines 146-178) + the `gelu_grad_method` A/B | when you need a number or the history of a design choice |
| `BOLTZ_MOE_BEST.md` | The two ~600-680M champion runs, layer-by-layer architecture, the 65B-token iso-token leaderboard vs Switch-MoE baselines | when scaling up or writing results. **Its "Source code" line refs are stale — see §10** |
| `TODO.md` | Queued work: sparsity sweep axes, τ sweep, repulsion sweep, scattermoe kernel, open questions | when picking the next experiment |
| `SCALE_UP_PLAN.md` | 3B and 7B param-count search, LR/token/GPU recommendations, FFN:Attn sanity checks | when scaling past 680M |
| `recovered_boltz_sessions/` | Recovered transcript of the purged pre-June session (179 msgs) — the Boltzmann-Hopfield design discussion lives here | archaeology only |

Do not re-derive results tables. `PROGRESS.md:146-178` is the master table;
`BOLTZ_MOE_BEST.md:83-105` is the large-scale comparison set.

---

## 1. What the project is

Replace the feedforward block of an Energy-GPT (EGPT) layer with a
**Boltzmann-weighted mixture of energy experts**. The energy is the
log-partition function over `K` parallel expert energies; the FFN output is its
gradient w.r.t. the hidden state:

```
E_MoE(h) = τ log Σ_k exp(E_k(h)/τ)
∇_h E_MoE = Σ_k p_k(h) ∇_h E_k(h),    p_k = softmax_k(±E_k(h)/τ)
```

The selling point: **routing needs no learned router.** The energy landscape
itself ranks the experts. `PROGRESS.md:127-132` records that at h1 scale this
matches or beats a learned-router top-k (0.501 vs 0.499 avg, 36.48 vs 39.79
PPL), and that energy-based selection survives hard top-2 truncation when
trained sparse.

Two expert families:

| Expert kind | Energy | Bounded below | Params per expert |
|---|---|---|---|
| `w1w2` | `E_k = -φ(W1_k h)·(W2_k h)` | no | `2·d·I_e` |
| `hopfield` | `E_k = (1/I_e)‖gelu(W_k h)‖²` | yes (≥0) | `d·I_e` |

---

## 2. Where things live

All paths relative to `/proj/dmfexp/nima/Code/dolomite-engine` unless absolute.

| Path | Contents |
|---|---|
| `configs/boltzmann_moe/` | 31 configs — the B/C/D/H series and the scale-ups (`h1_boltz_moe_580m_8x4096_d1536.yml`, `scale_h3_8gpt_4egpt_boltz_d1280.yml`, the 3B/7B picks) |
| `configs/multi_block_ablation/` | The FET / `math_fet_*` line, including the Hopfield and Boltz-Hopfield runs |
| `experiments/boltzmann-moe/results/` | 37 run dirs (B/C/D/H series + scale-ups). ~41 GB. gitignored |
| `experiments/boltzmann-moe/results/router_analysis/` | **new 2026-09-12** — router/FLOP analysis outputs |
| `experiments/boltzmann-moe/results/routing_cache/` | `routing_b{1-5}.pkl` cached routing vectors for the specialization PCA |
| `experiments/boltzmann-moe/scripts/` | run launchers, surrogate-router training, `watchdog/` auto-resubmit, **the two new 2026-09-12 analysis scripts** |
| `experiments/boltzmann-moe/paper/` | `report.tex`/`report.pdf` (10pp local report), `make_moe_scatter.py`, `figs/` |
| `experiments/energy-inference/results/multi-block-ablation/` | The `math_fet_*` checkpoints (Hopfield, Boltz-Hopfield, register variants). **This is where the 2026-09-12 target checkpoint lives, NOT under boltzmann-moe/results/** |
| `experiments/energy-inference/scripts/multi-block-ablation/` | `analyze_boltz_moe_routing_*.py`, `analyze_boltz_expert_{specialization,deep}_*.py`, eval/bench shell scripts |
| `~/Code/GPT-experiments/projects/EGPT-action/` | The FET/action project: `RESULTS.md`, `PLANS.md`, `SPECS.md`, `TODO.md`, on-policy verifier data under `results/onpolicy/` |

**Naming trap:** the H-series and B-series checkpoints are under
`experiments/boltzmann-moe/results/`, but the `math_fet_*` Boltzmann-Hopfield
checkpoints are under `experiments/energy-inference/results/multi-block-ablation/`.
The new `measure_moe_routing_20260912.py` resolves `--run` against the latter.

---

## 3. The two model-code lineages — know which one you are looking at

This is the single most confusing thing in the codebase. There are **two
independent implementations** of Boltzmann-MoE, and the checkpoints are split
across them.

### 3a. Legacy (pre-2026-06-28) — `mlp.py`

| Item | Location |
|---|---|
| Class | `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py:540` `BoltzmannMoE_Energy_MLP` |
| Config | `lm_engine/hf_models/config/mlp.py:136` `_BoltzmannMoEEnergyMLPArgs` |
| Dispatch | `lm_engine/hf_models/modeling_utils/mlp_blocks/__init__.py:113` |
| `mlp_type` | `BoltzmannMoE_Energy_MLP` |
| Experts | W1/W2 only (no Hopfield option) |
| Routing scale | `mlp.py:591` `_routing_scale = expert_I ** -0.5`, applied at `mlp.py:672` |
| top-k truncation | `mlp.py:682` `p = p * mask` — **no renormalization** |

Also in `mlp.py`: `BoltzmannMoE_Hopfield_Energy_MLP` (`mlp.py:244`) — a
standalone Hopfield-MoE that **lacks repulsion, τ and `n_repulsion_pairs`**.
Superseded by 3b; do not use for new runs.

Sibling classes worth knowing: `TopK_Energy_MoE_MLP` (`mlp.py:894`, learned
router + load-balance loss) and `SurrogateBoltzmannMoE_Energy_MLP`
(`mlp.py:994`) — the latter is prior art for a cheap inference router: it
trains a linear `d→K` head by KL against the Boltzmann weights and swaps it in
at eval (`use_surrogate=True`). Its own docstring notes the expert gradients are
still all computed, so it saves only the router.

**Checkpoints on this lineage:** all `b*`, `c*`, `d1_*`, `h1_*`, `h2_*`,
`scale_*` runs. e.g. `h1_boltz_moe_fullsize_d768` has
`mlp_type: BoltzmannMoE_Energy_MLP`, `n_experts: 4`, `intermediate_size: 8192`
(I_e=2048), `repulsion_coef: 0.1`, `temperature: 1.0`.

### 3b. Composable (2026-06-28 refactor) — `energy_ff.py`

`lm_engine/hf_models/modeling_utils/mlp_blocks/energy_ff.py`, 770 lines.

| Symbol | Line | Role |
|---|---|---|
| `FFEnergyBase` | 114 | abstract base; owns the `_capture_energy` / `_last_energy_per_token` FSDP-2 leaf-cache contract and `get_metrics()` |
| `W1W2FFEnergy` | 163 | `E = -(1/√I)·gelu(W1h)·(W2h)`; the `1/√I` moved **into the energy** (vs legacy, which applied `1/√I_e` at the routing layer only) |
| `HopfieldFFEnergy` | 259 | `E = (1/I)‖gelu(Wh)‖²`; single shared `W`, half the params of W1W2 |
| `BoltzmannMoEFFEnergy` | 338 | the MoE wrapper: routing, `e_sign`, repulsion, `top_k` |
| `_W1W2Expert` / `_HopfieldExpert` | 502 / 558 | view-backed experts, zero-copy row-slices of a fused weight |
| `_FusedW1W2Holder` / `_FusedHopfieldHolder` | 594 / 646 | own the actual `Parameter`s |
| `FusedMoEContainer` | 685 | what gets registered as `block.ffwd`; mirrors the inner MoE's `_last_energy_per_token` and `_cached_metrics` |
| `build_boltzmann_moe` | 718 | factory; picks `e_sign` per expert kind |
| Config | `config/mlp.py:245` `_EnergyFFBoltzmannMoEArgs` | `top_k` IS exposed (`config/mlp.py:264`) |
| Dispatch | `mlp_blocks/__init__.py:179` | passes `top_k` through (`__init__.py:191`) |
| top-k truncation | `energy_ff.py:432` `p = p * mask` | **no renormalization**, same as legacy |

`mlp_type` values: `EnergyFF_W1W2`, `EnergyFF_Hopfield`, `EnergyFF_BoltzmannMoE`
(+ `expert_kind: w1w2 | hopfield`).

`e_sign` convention (`energy_ff.py:426`): `"neg"` for Hopfield (route to the
**lowest** energy), `"pos"` for W1W2 (matches the validated legacy class).

**Checkpoints on this lineage:** the `math_fet_boltz_hopfield*` runs.

### 3c. The router-cost asymmetry — the key fact for any FLOPs work

```
Hopfield: E_k = (1/I_e)‖gelu(W_k h)‖²   → needs only W_k h  (FIRST matmul)
          ∇_h E_k = (4/I_e) W_kᵀ(gelu(W_k h)⊙gelu'(W_k h))  → SECOND matmul
```
The first matmul is **shared** between the router and the gradient. So for
Hopfield experts **routing is already free** — there is no separate router cost
to remove. The router costs exactly as much as all K expert-output matmuls
combined, but you have to pay it anyway.

```
W1W2:  E_k = h·term1_k,  term1_k = φ(W1_k h) @ W2_kᵀ
```
Here `term1` is itself a second matmul, so the router genuinely is an extra
matmul on top of `W1_k h`. **The "cheap proxy router" idea therefore pays off
roughly 2× more on the W1W2 lineage than on the Hopfield lineage.**

Consequence: an inference-FLOPs story built on "the energy router is expensive"
must be told on the **W1W2** line (`h1_boltz_moe_fullsize_d768`, the 680M), not
on the Hopfield line.

### 3d. How the FF plugs into the energy block

`lm_engine/hf_models/models/energy/layer.py`, class `EnergyBlock` (line 548),
main `forward` at line 792. For `energy_proj_type: psd_anti`:

```
ln_x    = ln(h)
grad_E  = attn_out + scale_ff · ffwd_out        # layer.py:885
h      := h - proj(grad_E)                       # Π = SSᵀ + (A - Aᵀ)
```
`ffwd_out = self.ffwd(ln_x)` at `layer.py:857`. `scale_ff` is a learned scalar
(init 4.0 by default, `layer.py:604`).

**The block is iterated.** `layer_iterations` (a per-layer list, consumed in
`lm_engine/hf_models/mixins/dense/base.py:257`) controls how many times each
block runs. For the 2026-09-12 target, block 8 runs **6×** — so the MoE cost is
multiplied by 6, and `p` is recomputed on every iteration.

---

## 4. Environment & ops

### Python env — avoid the flash-attn compile
`pip install -e .` on dolomite-engine triggers a 20-40 min flash-attn CUDA
build. For inference/analysis, borrow the nanoGPT venv and set `PYTHONPATH`:

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
```
New launchers should not hardcode either path. Source `experiments/paths.sh`, which self-locates
`REPO_ROOT` (so it is correct in any clone, from any cwd) and supplies `VENV` / `DATA_ROOT` with
this cluster's values as overridable defaults (`DOLOMITE_VENV`, `DOLOMITE_DATA_ROOT`).
`scripts/bsub/submit_train.sh` is the worked example. This does **not** apply to `configs/**` --
those paths are read by the trainer and are baked into saved checkpoint configs; see §4 gotcha 2.
Verified present: torch 2.12.0+cu130, transformers 4.46.3, accelerate 1.13.0,
safetensors 0.8.0, wandb 0.19.6. The energy model falls back to
`F.scaled_dot_product_attention` when flash-attn is absent.

### Cluster — IBM LSF
Check `hostname` first: `loginN.*` → login node, never run compute; `pN-rM-nK.*`
→ compute node, run directly. **A compute node does not imply an allocated
GPU** — check `nvidia-smi`; if it reports "No devices were found", you still
need `bsub` for GPU work.

```bash
mkdir -p ~/bsub_logs      # always log here, never $HOME directly
bsub -q preemptable -G grp_preemptable -J <name> \
     -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 02:00 \
     -o "$HOME/bsub_logs/<name>_%J.stdout" \
     -e "$HOME/bsub_logs/<name>_%J.stderr" < script.sh
```
`-G grp_preemptable` is required with `-q preemptable`. Verify 30-60 s after
submission that `STAT=RUN`, not `EXIT`.

Multi-node: `scripts/common/pretrain.sh` reads `LSB_MCPU_HOSTS` /
`LSB_JOBID` / `CUDA_VISIBLE_DEVICES` to derive master addr/port, NNODES and
GPUS_PER_NODE automatically.

### Gotchas that will bite you

1. **`max_to_keep: 2`** — always set it in `save_args`. The field is
   `int | None` and pydantic rejects `null` when reloading a saved checkpoint
   config. (`lm_engine/arguments.py`, `SaveArgs`.)
2. **Auto-resume needs an explicit `load_args.load_path`.** Without it training
   silently restarts from scratch even though a checkpoint exists. See the
   project memory note `feedback_dolomite_resume.md`.
3. **`export TMPDIR=<writable path>` in every launcher.** See §7.1.
4. **Watchdog**: `experiments/boltzmann-moe/scripts/watchdog/` auto-resubmits on
   preemption. `watchdog_jobs.conf` sets `max_resubmits`; reset the counter in
   `watchdog_state.txt` if a run hits the cap.

---

## 5. Results — the numbers that drive decisions

Full tables: `PROGRESS.md:146-178` and `BOLTZ_MOE_BEST.md:83-105`. Carry these
four rows in your head:

| Row | Avg | WikiPPL | Why it matters |
|---|---:|---:|---|
| `h1_boltz_moe_fullsize` (soft, K=4×2048) | 0.501 | 36.48 | the W1W2 reference; Boltzmann ≥ learned top-k |
| `h1_topk_egpt_moe` (learned router, top-2) | 0.499 | 39.79 | the learned-router control |
| `h1_boltz_topk2` — **trained** sparse top-2 | 0.486 | **36.37** | sparse training costs ~nothing on PPL |
| `h1_boltz_full @ top2 eval` — **post-hoc** truncation | 0.489 | **42.84** | post-hoc truncation costs **+6.4 PPL** |

**The single most important asymmetry in the project:** training with top-k is
free; truncating a soft-trained model to top-k at eval costs +6.4 PPL. Part of
that is almost certainly the missing renormalization — both implementations do
`p = p * mask` so the retained mass sums to `< 1`, shrinking the output
magnitude (`energy_ff.py:432`, `mlp.py:682`). **A top-k-with-renormalization A/B
on an existing checkpoint is a cheap, unrun experiment.** `TODO.md:107-111`
lists this as an open question with a proposed diagnostic.

Large scale (`BOLTZ_MOE_BEST.md`): at 65B iso-tokens the 680M Boltz
(679M/679M active, energy-attn + recurrent EGPT) reaches avg 58.47 / PPL 19.33,
but the matched-structure Switch ablation reaches 57.82 / 19.81 and a larger
pure-GPT Boltzmann model (962M/585M) reaches 58.02 / 19.73. The honest
conclusion recorded there: architecture advantages are **real but modest**
(~0.5-0.7pp avg, ~0.5 PPL), and 1.4× more params can substitute for them.

---

## 6. Overleaf targets

| Repo | Remote | Boltz-MoE content |
|---|---|---|
| **`~/Code/overleaf/boltzmann-moe-ICLR-2026/`** | Overleaf git-bridge `origin` | **★ ACTIVE PAPER.** `main.tex` + `sec/{intro,theory,experiments,appendix}.tex`. Tables: `tab:frontier`/`tab:pure`/`tab:cost`/`tab:threeway` (experiments), `app:frontier`/`app:eval`/`app:scale400`/`app:threeway`/`tab:attribution` (appendix). **All `Avg11` since 2026-09-15.** |
| `~/Code/energy/energy-GPT-neurips2026/` | Overleaf git-bridge `origin` | **ARCHIVE — still `avg10`, do not read unless asked.** `nima/sec/appendices/boltz_moe.tex` (37 KB), included from `nima/paper_v2.tex:108` |
| `~/Code/overleaf/energy-GPT-reformulation-2026/` | Overleaf git-bridge `origin` | **ARCHIVE — `avg10`/`avg9`.** `moe_bs.tex` at top level + `slides/`, talk frames with the MoE results table and scatter plot |

Local report: `experiments/boltzmann-moe/paper/report.pdf`. Scatter figures are
generated by `paper/make_moe_scatter.py` into `paper/figs/`.

---

## 7. SESSION FINDINGS — 2026-09-12

**Target checkpoint throughout this section:**
```
math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu
config: configs/multi_block_ablation/<same name>.yml
ckpt:   experiments/energy-inference/results/multi-block-ablation/<same name>/unsharded/
```
Architecture: `d=1536`, 9 layers = 8 GPT (softmax attn 24 heads, swiglu
I=4096) + 1 energy block (energy_attention 24 heads, `EnergyFF_BoltzmannMoE`
`expert_kind=hopfield`, K=8, I_total=8192 → I_e=1024, τ=1.0,
`repulsion_coef=0.01`, `gelu_grad_method=sigmoid`), block 8 iterated **6×**,
`iter_dropout_range_per_block[8]=3`, `energy_proj_type=psd_anti`,
`energy_antisym_rank=32`, `energy_apply_rayleigh=true`, tied embeddings,
vocab 100352. 400.4M params (800.8 MB bf16 — file size confirms exactly).
Trained to step 32000 ≈ 33.5B tokens.

Reported evals for this checkpoint (from the JSONs in `unsharded/`):
wikitext word-PPL **26.03**, sciq 0.853, piqa 0.690, arc_easy 0.599,
hellaswag acc_norm 0.417, mmlu 0.2785, BBH mean over 28 tasks **0.2771**,
gsm8k flexible-extract 2.27%.

### 7.0 Provenance & confidence

| Finding | Basis | Confidence |
|---|---|---|
| FLOP breakdown (§7.2) | exact arithmetic over checkpoint tensor shapes | **final** |
| Expert SVD spectra (§7.3) | exact, from the weights | **final** |
| Routing uniformity (§7.4) | real BBH activations, but **CPU smoke test, N=4-6 prompts** | **provisional**, corroborated by an independent magnitude argument |
| Dead FF branch (§7.5) | real activations + paired-NLL ablation, **N=4-6 prompts / 355-545 tokens** | **provisional but very large effect** |
| Cheap-router agreement (§7.6) | 240 cached real activations, post-bugfix | **provisional** |
| Iteration stability (§7.7) | 359 tokens/iter | **provisional** |

**A full-scale confirmation run is in flight: bsub job `1559723`** (1 GPU, 810
prompts × 1024 tokens × 3 checkpoints, with `--ablate --n_ablate 250`). At the
time of writing it was `RUN` on `p3-r25-n2`, ~635 s in, still on the first
checkpoint. **If you are reading this later, check for the results first:**

```
experiments/boltzmann-moe/results/router_analysis/routing_measure__<run>.json
```
for the three runs `math_fet_boltz_hopfield_rep_*`,
`math_fet_boltz_hopfield_*` (no-repulsion sibling), and
`math_fet_hopfield_mean_*`. Also `~/bsub_logs/moe_router_measure_1559723.stdout`.
Note that `router_analysis/routing_measure_20260912.json` (old filename, no
`__<run>` suffix) is a **stale CPU smoke-test artifact** from before the
`--run` flag existed — safe to delete.

### 7.1 Folder cleanup (~190 stray temp dirs) — root cause found

`experiments/boltzmann-moe/` had grown to 201 entries, ~190 of which were dead
temp directories. **None were Claude Code artifacts.**

| Removed | Count | What it was |
|---|---:|---|
| `pymp-*/` | 113 | `multiprocessing` scratch; each held a dead `listener-*` unix socket |
| `tmp*wandb-media/`, `tmp*wandb-artifacts/` | 64 | wandb upload staging, all **empty** |
| `torchelastic_*/` | 5 | `torchrun` elastic-agent scratch (`attempt_0/<rank>`) |
| `tmp<8char>/` | 6 | `torch.distributed.rpc` scratch (`_remote_module_non_scriptable.py`) |
| `torchinductor_ndehmamy/` | 1 | empty `torch.compile` cache |

**Root cause:** Python's `tempfile` falls back to the **cwd** when `TMPDIR` is
unset and `/tmp` is unwritable — which is the case inside these LSF jobs. The
runs were launched with `cwd = experiments/boltzmann-moe` (confirmed:
`wandb/run-*/files/wandb-metadata.json` has
`root = .../experiments/boltzmann-moe`). `TMPDIR` was verifiably unset.

**Fix:** `export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "$TMPDIR"` in
every launcher. Patterns were also added to
`experiments/boltzmann-moe/.gitignore` so they never show up in `git status`
again.

### 7.2 Exact inference-FLOP breakdown and the top-k ceiling

Script: `scripts/analyze_moe_router_flops_20260912.py` (CPU). Counts weight
matmul MACs exactly from the checkpoint's tensor shapes; attention score/AV
terms are sequence-length dependent and excluded.

```
per-token matmul MACs (dense soft routing)
  gpt_layers_x8            226.49 M   37.5%
  egpt_attn_x6              42.47 M    7.0%
  egpt_proj_x6              29.49 M    4.9%
  egpt_moe 1st matmul x6    75.50 M   12.5%   <- W h: gives BOTH E_k and the pre-activations
  egpt_moe 2nd matmul x6    75.50 M   12.5%   <- gated_k @ W_k, per expert
  lm_head                  154.14 M   25.5%
  TOTAL                    603.59 M   = 1.207 GFLOP/token
```

Savings ceiling. "Exact energy router" must still compute `W h` for all experts
(so it only saves on the second matmul); a "free router" lets you skip the
first matmul for unselected experts too:

| variant | MoE | EGPT block | whole model |
|---|---:|---:|---:|
| dense soft (current) | 1.000 | 1.000 | 1.000 |
| top-4, exact energy router | 0.750 | 0.831 | 0.937 |
| top-2, exact energy router | 0.625 | 0.746 | **0.906** |
| top-1, exact energy router | 0.562 | 0.704 | 0.891 |
| top-4, free router | 0.500 | 0.661 | 0.875 |
| top-2, free router | 0.250 | 0.492 | **0.812** |
| top-1, free router | 0.125 | 0.407 | **0.781** |

**Read this carefully: up to −59% on the EGPT block but only −22% end-to-end**,
because the LM head (25.5%) and the 8 GPT layers (37.5%) dominate. top-k alone
buys 9%; the cheap router is what unlocks the rest. §9 discusses why −22% is
arguably the wrong denominator for a method claim.

### 7.3 The expert weights are near rank-2

SVD of each `W_k` (`[1024, 1536]`), from `analyze_moe_router_flops_20260912.py`.
`eff_rank` = participation ratio of the squared spectrum,
`(Σσ²)² / Σσ⁴`. Columns are the cumulative fraction of `‖W‖_F²` captured:

```
  k   ||W||_F  eff_rank   r=1    r=2    r=8    r=16   r=256
  0     7.316       2.0   0.674  0.908  0.932  0.938  0.972
  1     7.542       2.9   0.537  0.759  0.887  0.927  0.974
  2     7.159       2.6   0.596  0.776  0.887  0.929  0.971
  3     7.280       2.2   0.606  0.903  0.929  0.935  0.971
  4     7.387       2.2   0.629  0.886  0.928  0.937  0.972
  5     7.064       1.9   0.714  0.867  0.921  0.932  0.970
  6     7.100       2.2   0.632  0.855  0.920  0.933  0.970
  7     7.306       2.2   0.607  0.895  0.929  0.935  0.970
```

- Effective rank **1.9-2.9 out of 1024**. Rank-16 captures ~93%.
- `‖W‖_F` total decayed from **70.9 at init** (`0.02·√(8192·1536)`) to **20.5**.
  For scale, the GPT layers' `c_fc` have `‖·‖_F ≈ 183-190`.
- Cross-expert redundancy, `cos(W_iᵀW_i, W_jᵀW_j)`: mean **0.184**, min 0.021,
  max 0.620 — the experts are geometrically distinct, so routing has real
  content available to it in principle.

Since `W_k` is effectively rank-2, `E_k(x) = F_k(V_kᵀ x)` with `V_k ∈ ℝ^{d×r}`,
`r ≈ 2-8`. That is the whole basis for the cheap router in §7.6. It also
suggests the experts themselves are massively over-parameterized — a separate,
unexplored compression opportunity.

### 7.4 Routing is uniform

On real BBH activations (`measure_moe_routing_20260912.py`, section A):

```
  iter    tokens   eff_n    top1    top2    top4     E_mean    E_range    |out|
     0       359   7.999  0.1264  0.2527  0.5045    0.01360    0.03413    0.005
     ...
     5       359   7.999  0.1266  0.2530  0.5051    0.01473    0.04013    0.005
  (uniform reference: eff_n=8, top1=0.1250, top2=0.2500, top4=0.5000)
```

`effective_n_experts = 7.999 / 8`. Top-1 mass 0.1266 vs the uniform 0.1250.

**Cause:** `E_k ≈ 0.003-0.015` against `τ = 1.0`. The `(1/I_e)` MEAN
normalization in the Hopfield energy, compounded by weight decay shrinking
`‖W‖_F` 3.5×, puts the energies **2-3 orders of magnitude below the
temperature**, so `softmax(-E_k/τ)` is flat by construction.

An independent order-of-magnitude check reproduces this from the weights alone
(σ₁ ≈ 6, ‖ln_x‖ ≈ 19, energy spread over 1024 units with a `1/I_e` mean →
E ≈ 0.003), so the conclusion does not depend on the small prompt sample. The
one caveat that *could* have overturned it — real activations aligning with an
expert's top singular direction, which would raise `E_k` by up to ~40² — is
exactly what the real-activation measurement rules out.

Sharpening sweep on cached real activations (section C, frozen weights):

| scheme | eff_n | top1 | top2 |
|---|---:|---:|---:|
| τ=1.0 (as trained) | 7.999 | 0.1268 | 0.2533 |
| τ=0.1 | 7.905 | 0.1423 | 0.2822 |
| τ=0.03 | 7.293 | 0.1781 | 0.3463 |
| τ=0.01 | 5.754 | 0.2537 | 0.4673 |
| τ=0.003 | 3.848 | 0.4417 | 0.6931 |
| **τ=0.001** | **1.994** | 0.7325 | 0.9168 |
| z-score per token | 6.528 | 0.2157 | 0.4093 |

So τ ≈ 1e-3 is what "sharp" costs here. Note that **per-token z-scoring alone
is not enough** (eff_n 6.5) — normalizing K=8 logits to unit std still leaves a
fairly flat softmax. A z-score **plus a learned gain** (≈3×) is the natural
scale-free fix, and is the direct analogue of the `1/√expert_I` routing-scale
fix that rescued the B-series (`PROGRESS.md:110`, `PROGRESS.md:155-156`).

Per-task routing profiles (section F) showed **no specialization**: max
across-task swing in any expert's mean weight was 0.0021, i.e. every one of the
sampled BBH tasks routes essentially uniformly. (Provisional — 4 tasks in the
smoke test; the full run covers 27.)

### 7.5 **The energy-FF branch is essentially dead** ← headline

`grad_E = attn_out + scale_ff · ffwd_out` (`layer.py:885`). Measured branch
magnitudes (section A2):

| quantity | Boltz-MoE-Hopfield (target) | non-MoE `hopfield_mean` sibling |
|---|---:|---:|
| mean `‖h‖` (residual stream into block) | 433.49 | 431.81 |
| mean `‖attn_out‖` | 24.34 | 22.34 |
| mean `‖ffwd_out‖` | **0.0054** | **0.5469** |
| `scale_ff` (learned) | 1.5156 | 0.5156 |
| mean `‖scale_ff · ffwd_out‖` | **0.0082** | **0.2820** |
| **FF share of `grad_E` magnitude** | **0.03%** | **1.25%** |
| FF perturbation relative to `‖h‖` | 0.0018% | 0.0653% |

Paired-NLL ablation on **identical tokens** (section G — same tokens, same
deterministic model, so the *difference* has no sampling error; only
representativeness of the prompt set is at issue). **Full scale, 87,997 tokens**
(job 1559723; the earlier 545-token smoke-test figures are superseded):

| variant | FET-rep (Hopfield K=8) | `hopfield_mean` (no MoE) | h1_fullsize (w1w2 K=4) | h1_topk2 (sparse-trained) |
|---|---|---|---|---|
| baseline ppl | 5.9375 | 5.9645 | 7.6881 | 7.5838 |
| **branch DELETED** (`ffwd_out=0`) | **+0.0003** | +0.0033 | **+10.76** | **+10.36** |
| routing forced UNIFORM | +0.0009 | n/a | +1.94 | +1.43 |
| top-4 (truncated, no renorm) | −0.0009 | n/a | +0.0000 | +0.1201 |
| top-2 | −0.0010 | n/a | +0.68 | **+0.0000** |
| top-1 | +0.0002 | n/a | +2.40 | **+0.65** |

**Deleting the entire MoE branch changes FET-rep's perplexity by +0.0003 —
indistinguishable from zero.** The w1w2 line is the opposite: +10.8 ppl to delete,
+1.9 ppl to flatten routing. And note the **sparse-training asymmetry**: the
sparse-trained model pays *exactly zero* for top-2 and only +0.65 for top-1,
where the soft-trained one pays +0.68 / +2.40. Training sparse makes top-k free;
truncating post-hoc does not.

Four consequences:

1. **For this checkpoint the largest inference-FLOP win is deleting the MoE
   outright: −151.0M of 603.6M MACs/token = −25.0% total FLOPs, −12.6M params,
   zero measurable quality cost.** That is larger than the −22% best case from
   top-k + a free router, and it needs no finetuning at all.
2. **It explains the uniform routing.** With the branch output irrelevant to
   the loss there is no gradient pressure to differentiate experts; weight
   decay was free to shrink `W` 3.5× and drive it to rank ~2.

   **ROOT CAUSE, measured (section A3, job 1560005 — read this before blaming
   the repulsion loss).** An earlier reading of this session attributed the dead
   branch to the repulsion term. That is WRONG, and the expert-output geometry
   says so:

   | | FET-rep (K=8) | h1_fullsize (K=4) | h1_topk2 (K=4, sparse) |
   |---|---:|---:|---:|
   | `repulsion_coef` | 0.01 | **0.1 (10×)** | 0.1 |
   | mean pairwise `cos(gᵢ,gⱼ)` | −0.1394 | −0.3207 | −0.2308 |
   | geometric floor `−1/(K−1)` | −0.1429 | −0.3333 | −0.3333 |
   | fraction of the way to the floor | 97.6% | 96.2% | 69.2% |
   | cancellation `‖Σpg‖/Σp‖g‖` | 0.241 | **0.190** | **0.663** |
   | mean `‖g_k‖` | **0.019** | **155.66** | 72.18 |
   | `E` mean / cross-expert spread | 0.0126 / 0.0117 | 0.0965 / 0.4216 | 1.356 / 1.417 |
   | `effective_n_experts` | 7.999 / 8 | 3.746 / 4 | 2.413 / 4 |
   | FF share of `grad_E` | 0.04% | 85.84% | 80.87% |

   h1 did **not** avoid the anti-alignment (it is equally pinned to the geometric
   floor, at 10× the repulsion coefficient) and did **not** avoid the cancellation
   (it loses *more*: 19.0% surviving vs 24.1%). Cancellation costs both ~5×, so it
   is not the discriminator. The accounting closes as
   `‖ffwd_out‖ ≈ mean‖g_k‖ × cancellation`: 0.019×0.241 = 0.0046 (measured 0.0050)
   and 155.66×0.190 = 29.6 (measured 25.0).

   The discriminator is `mean ‖g_k‖`, 0.019 vs 155.66 — **8,200×** — and 256× of
   that is a bare structural constant in the source:

   | class | output prefactor | value at `I_e=1024` |
   |---|---|---|
   | `_HopfieldExpert` (`energy_ff.py:644`) | `4/I_e` | **0.0039** |
   | `_W1W2Expert` (`energy_ff.py:604`) | `I_e^-0.5` | 0.031 |
   | legacy `BoltzmannMoE_Energy_MLP` (`mlp.py:693`) | **none** | **1.0** |

   `scale_ff` (1.5156) cannot bridge 4 orders of magnitude; it would need ~8000.
   So the real mechanism is a **feedback loop whose sign depends on whether the
   branch is load-bearing at initialization**: the `4/I_e` prefactor makes the
   Hopfield branch weak from the start → its output barely moves the loss → weak
   gradient signal → repulsion's anti-alignment plus cancellation weakens it
   further → weight decay wins → collapse. On h1 the branch is 86% of `grad_E`, so
   the loss actively needs it and the same repulsion at 10× strength only reshapes
   geometry without shrinking magnitude. Repulsion is an **accelerant, not the
   cause** — fixing it alone will not revive the Hopfield-MoE line.

   Two independent defects, both from `1/I_e`: (a) the `4/I_e` **output** prefactor
   kills the branch magnitude; (b) the `1/I_e` **energy** scale leaves `E ≈ 0.013`
   against `τ=1`, giving uniform routing and making top-k a pure loss. Fix (a) with
   `4/I_e → 1/√I_e` on the output; fix (b) with `routing_norm` (see §7.10).
3. **The whole Hopfield-MEAN energy-FF line has a weak-to-dead FF branch**, and
   the MoE variant is ~36× worse than the plain one. The `(1/d_int)` MEAN form
   was introduced to stop the descent step growing with `intermediate_size`
   (NaNs in EGPT-RL run 1714840 — see the `HopfieldFFEnergy` docstring,
   `energy_ff.py:259-270`); at these widths it over-suppresses by ~3 orders of
   magnitude.
4. **The reported MoE-vs-Hopfield-MEAN delta was comparing two near-dead
   branches.** The config header of
   `configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml` cites
   0.5073 (Boltz-MoE) vs 0.5135 (Hopfield-MEAN) avg10_norm and attributes the
   deficit to narrow experts + MEAN scale-invariance. The real mechanism is
   that neither FF branch is load-bearing, and the MoE's is 36× weaker.
   (Metric note, 2026-09-14: those 0.5073 / 0.5135 are legacy `avg10_norm`; both
   `math_fet_*` checkpoints predate the `pyarrow>=20` fix and are **INCOMPLETE (9/11)**
   under the canonical `compute_avg11.py`, so they cannot be restated to Avg11 without
   re-evaluating. The −0.62pp delta is avg10-vs-avg10 and stays valid as a relative
   comparison.)

Note that under exactly uniform `p`, the MoE is **algebraically identical** to
a plain dense Hopfield FF of width `I_total`:
`Σ_k (1/K)(4/I_e)(gated_k @ W_k) = (4/I_total)(gated @ W_fused)`. The ablation
row "routing forced UNIFORM" (Δ −0.0040) confirms this empirically. So even if
you keep the branch, it can be collapsed to one fused GEMM instead of 8 sliced
ones — same FLOPs, better GEMM utilization, no routing.

### 7.6 Cheap-router feasibility on real activations

Section D, scored by **row-wise top-k selection agreement** with the exact
energy router (what actually matters — not energy MSE):

| variant | r | top1 | top2 | top4 | MACs/token |
|---|---:|---:|---:|---:|---:|
| spectral `‖W_k x‖²` | 2 | 0.0000 | 0.0000 | 0.2281 | 24.6 K |
| **exact-in-rank-r subspace** | 2 | 0.7792 | 0.6812 | 0.7937 | 24.6 K |
| spectral `‖W_k x‖²` | 8 | 0.0000 | 0.0021 | 0.2542 | 98.3 K |
| **exact-in-rank-r subspace** | 8 | **0.9375** | **0.8958** | 0.9323 | **98.3 K** |
| exact router (reference) | — | 1.0 | 1.0 | 1.0 | **12.58 M** |

Two results:

- **The naive `½‖W_k x‖²` proxy FAILS** (~0% top-1 agreement). The reasoning
  "GELU ≈ ReLU, and `E‖relu(z)‖² = ½‖z‖²`, so route on `‖W_k x‖²`" does not
  survive contact with these weights: `gelu(z)²` is not proportional to `z²`
  pointwise, and with `e_sign="neg"` (route to the *lowest* energy) the
  induced ordering is essentially uncorrelated. Confirmed on synthetic input
  too, where even a **full-rank** `‖Wx‖²` proxy topped out at 0.68 top-1 — so
  the failure is the nonlinearity, not the truncation.
- **Restricting the exact energy to the rank-r subspace works.** At r=8:
  0.94 top-1 / 0.90 top-2 for 98.3 K MACs vs the exact router's 12.58 M —
  a **~128× cheaper router**.

> **⚠ CORRECTION 2026-09-16 — READ THIS BEFORE REUSING THE 0.94/0.90 ABOVE.**
> Those numbers are for the **exact energy restricted to the rank-r subspace**, i.e.
> `mean(gelu(W_k V_r V_rᵀ x)²)`, which keeps the true nonlinearity. They are **NOT** measurements
> of the "tiny fitted nonlinear head" recommended just below, and they were subsequently quoted as
> if they were — in `energy_ff.py`'s proxy comment, in HANDOFF §12.9, and in my own reasoning.
> Measured on `pure_hop_T12_sink`, the fitted diagonal-quadratic head reaches **0.348 top-2 at
> r=8** against a chance floor of `k/K` = **0.125**, and only 0.461 at r=32. A diagonal quadratic
> in `a` cannot represent `mean(gelu(·)²)`. See §12.11. The cost line above is also incomplete:
> `K·d·r` counts only the projection and omits the `K·I_e·r` term for `B_k a_k`.

**Recommended implementation:** project `a = V_kᵀ x` (r ≈ 2-8 coefficients,
cost `d·r` per expert) then evaluate a **tiny fitted nonlinear head** on `a`.
Because `r` is only 2-8, a low-degree polynomial or a 16-unit MLP suffices and
is essentially free. Fit it with a **frozen backbone** by regression / KL
against cached `(x, E_k)` pairs — exactly what
`router_fit_cache__<run>.pt` contains. `V_k` initializes for free from the SVD
of the trained `W_k`; no training run required.

### 7.7 Iteration stability — route once, reuse 6×

Section B, comparing `p` at iteration *t* against iteration 0 of the same
forward pass:

| iter | argmax agree | total variation |
|---:|---:|---:|
| 1 | 0.9109 | 0.0008 |
| 2 | 0.8997 | 0.0009 |
| 3 | 0.8969 | 0.0010 |
| 4 | 0.8942 | 0.0011 |
| 5 | 0.8858 | 0.0011 |

Routing is nearly identical across the 6 iterations, so it could be computed
**once** and reused — a 6× router cut with no proxy and no retraining.

**Caveat that must not be lost:** this is measured in the degenerate
uniform-routing regime, where *everything* is trivially "stable". **Re-measure
once routing is actually sharp** before claiming this.

### 7.8 Bug fixed: no `EnergyFF_*` run ever logged its routing metrics

`lm_engine/train_utils.py` gated the energy-MLP metric collection on
`isinstance(module, (Energy_MLP, Compositional_Energy_MLP, Mixed_Energy_MLP,
BoltzmannMoE_Energy_MLP))` — which **omits the entire `FFEnergyBase` family**.
Result: no run using `EnergyFF_W1W2` / `EnergyFF_Hopfield` /
`EnergyFF_BoltzmannMoE` ever logged `effective_n_experts`,
`n_dominant_experts`, `max_expert_load` or `mean_token_entropy_norm`.

Verified on the target run: `wandb-summary.json` has 17 keys, and the only
`model/energy_mlp/*` entries are three `attn` norms (`W_Q_norm`,
`c_attn_norm`, `output_norm`). No routing metrics at all.

**Fix applied** (`lm_engine/train_utils.py:73-87`): added `FFEnergyBase` to the
isinstance tuple, with a `name.rsplit(".", 1)[-1] != "moe"` guard because
`FusedMoEContainer` mirrors its inner `BoltzmannMoEFFEnergy`'s metrics and
would otherwise emit every series twice. Verified: exactly one series per FF
module (`ffwd` logs, `ffwd.moe` is skipped).

**Consequence for readers of old docs:** the "experts collapsing to uniform
routing" claim in the header comment of
`configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml:12` was
**inferred, never measured**. As of §7.4 it happens to be correct — but for a
different reason than the one stated there (energies ≪ τ, not narrow experts).

### 7.9 Methodology warning: `torch.isin` silently fakes top-k agreement

`torch.isin(elements, test_elements)` **flattens** `test_elements`. So

```python
a = E.topk(k, -1).indices          # [N, k]
torch.isin(b[:, j], a)             # WRONG: compares against ALL N rows at once
```

compares each row's candidate against the union over the whole batch. With
`N=240` and `k=4` the flattened set contains nearly every expert index, so this
reports **~1.0000 agreement for anything**, including a proxy that is actually
0% correct. This produced fake `top2_overlap = 1.0000` and `top4 = 1.0000`
numbers before it was caught.

Correct row-wise form — `_row_overlap` in
`scripts/measure_moe_routing_20260912.py:56`:

```python
a = ref.topk(k, -1, largest=largest).indices
b = cand.topk(k, -1, largest=largest).indices
mask = torch.zeros_like(ref, dtype=torch.bool).scatter_(-1, a, True)
return mask.gather(-1, b).float().mean().item()
```

Any top-k overlap / routing-agreement metric anywhere in this project should be
audited against this. Note the numbers in §7.6 are post-fix; the pre-fix run
reported `exact-in-rank-8` as 0.9792/0.9729/1.0000 instead of the true
0.9375/0.8958/0.9323.

---

### 7.10 Fixes landed 2026-09-12 (code changes, not just findings)

| fix | where | default | note |
|---|---|---|---|
| `FFEnergyBase` added to the metrics isinstance gate | `train_utils.py:73` | — | no `EnergyFF_*` run had ever logged routing metrics; deduped with `name.rsplit(".",1)[-1] != "moe"` because `FusedMoEContainer` mirrors its inner wrapper |
| `repulsion_form ∈ {squared, abs, hinge, signed}` | `energy_ff.py::_repulsion_penalty`, legacy `mlp.py::_add_repulsion_loss`, config `_EnergyFFBoltzmannMoEArgs` | **`squared`** | legacy `signed` is minimised at `cos=-1` and rewards anti-alignment. **λ recalibration:** at \|cos\|≈0.3, `squared` is ~3× weaker than `signed`/`abs` (0.09 vs 0.30) because cos² vanishes quadratically near orthogonality — **`abs` is the drop-in that preserves the existing λ sweeps** (0.1 for B4/B5, 0.01 for FET); `squared` wants λ scaled up. Re-sweep. |
| `routing_norm ∈ {none, zscore, sqrt_width}` | `BoltzmannMoEFFEnergy._logits` | `none` | decouples the routing logit scale from the gradient scale, which in this class are the SAME object. On untrained Hopfield weights (K=8): `none` → `eff_n` 7.999, `sqrt_width` → 6.900, `zscore` → 5.580. `zscore` normalises to unit std, so pair it with `temperature ≈ 0.35` to reach `eff_n ≈ 2`. |
| `head_dim` decoupled from `hidden_size/num_heads` | `energy_attention.py` (`qk_dim = num_heads*head_dim`; three `c_attn.weight` slices; `add_wv_wo` assert), config `_EnergyAttentionArgs`, dispatcher | `None` = coupled | enables over-complete heads (`num_heads*head_dim > d`). Verified fwd+bwd at `D_qk = 4d`; partial RoPE already supported (`rope.py:118`). Both existing checkpoint families load with identical shapes. |

### 7.11 Measured prefill throughput (jobs 1559949/1559955/1559998/1560169)

`scripts/bench_moe_throughput_20260912.py`, H100 80GB, 32,768-token prefill,
`7b_K128` (d=2560, K=128, top_k=8):

| path | ms/call | µs/token | speedup | TFLOP/s | MAC ratio | chunk |
|---|---:|---:|---:|---:|---:|---:|
| `dense-soft` | 2421.83 | 73.91 | 1.00× | 360.9 | 1.0000 | 512 |
| `topk-masked` **(what ships)** | 2425.18 | 74.01 | **1.00×** | 360.4 | 1.0000 | 512 |
| `topk-grouped-exact` | 2027.77 | 61.88 | 1.19× | 229.0 | 0.5312 | 1024 |
| `topk-grouped-proxy` (Python loop) | 127.51 | 3.89 | **18.99×** | 431.2 | 0.0629 | 32768 |
| `topk-gmm-proxy` (`torch._grouped_mm`) | 140.45 | 4.29 | 17.24× | 391.4 | 0.0629 | 32768 |

(Job 1560169, with the weight transpose hoisted. Job 1559998 measured the gmm path
at 12.88× because it transposed the 13.3 GB expert tensor *inside* the timed
region — a bug in the first version of this benchmark, not a property of
`_grouped_mm`. Disregard any 12.88× / 6.78× figure.)

Three things to carry forward:

1. **`top_k` buys exactly 0% wall-clock today** — `p = p * mask` (`energy_ff.py:432`,
   and the legacy class) computes every expert then zeroes the rejected ones.
2. **The fix is batch size, not a kernel.** The sparse path must see large token
   batches: at a 512-token chunk each of K=128 experts gets 32 rows and the loop
   manages only 2.63×; at the full 32,768-token batch each gets 2048 rows and the
   plain per-expert **Python loop reaches 19.02× at 431.7 TFLOP/s** — at/above the
   15.9× arithmetic ceiling (it exceeds it because `dense-soft` is itself forced to
   chunk). Nothing is left for a fused kernel to recover.
3. **`torch._grouped_mm` matches but does not beat the loop** once its weight
   transpose is hoisted out of the timed region. Loop vs gmm, consistently ~2–10%
   on the loop's side:

   | shape | tokens | loop | gmm |
   |---|---:|---:|---:|
   | `3b_K32` | 8192 | 9.90× | 9.63× |
   | `3b_K32` | 32768 | 10.35× | 9.36× |
   | `7b_K64` | 32768 | 10.31× | 8.53× |
   | `7b_K128` | 8192 | 16.81× | 16.57× |
   | `7b_K128` | 32768 | 18.99× | 17.24× |

   The residual deficit is the `[N·k, d]` / `[N·k, I_e]` gather/scatter the loop
   avoids by slicing. Expect `_grouped_mm` to win in the **opposite** regime — small
   batches, decode, or very large K, where per-expert M is small and launch overhead
   dominates (at a 512-token chunk the loop managed only 2.63×). Rule of thumb:
   large batch → per-expert loop; small batch or large K → `_grouped_mm`.

Also: **`dense-soft` cannot run un-chunked at all** at production width — `W h` is
`[32768, 2.6M]` = 171 GB in bf16. The sparse path never forms it. That is an
activation-memory argument for sparse routing independent of FLOPs.

## 8. New scripts (2026-09-12)

Both live in `experiments/boltzmann-moe/scripts/`.

### `analyze_moe_router_flops_20260912.py` — CPU only, seconds

Exact per-token FLOP model derived from the checkpoint's own tensor shapes
(not a hand-written parameter count), plus per-expert SVD spectra and the
cross-expert Gram cosine.

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
OMP_NUM_THREADS=4 python experiments/boltzmann-moe/scripts/analyze_moe_router_flops_20260912.py
#   --ranks 1 2 4 8 16 32 64 128 256     which cumulative-energy ranks to report
#   --skip-spectra                        FLOPs only
```
The checkpoint is currently hardcoded (`CKPT` at the top of the file) — edit it
or parameterize if you point it elsewhere.

### `measure_moe_routing_20260912.py` — needs a GPU

The Phase-0 instrumentation. Runs real BBH few-shot prompts through the model,
hooks the MoE (or the plain FF), and reports everything in §7.4-§7.7.

```bash
python experiments/boltzmann-moe/scripts/measure_moe_routing_20260912.py \
    --run math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu \
    --n_prompts 810 --max_len 1024 --cache_per_iter 6000 \
    --ranks 1 2 4 8 16 32 64 --ablate --n_ablate 250
```

| flag | meaning |
|---|---|
| `--run` | run dir name under `experiments/energy-inference/results/multi-block-ablation/` |
| `--n_prompts` | prompts to sweep for routing stats (spread evenly over the 27 BBH task files) |
| `--max_len` | truncation length |
| `--cache_per_iter` | `(x, E_k)` pairs cached **per iteration** for the frozen-backbone router fit |
| `--ranks` | ranks to test in the cheap-router agreement table |
| `--ablate` | run section G (paired-NLL branch ablation) |
| `--n_ablate` | prompts for the ablation; note it does 6 passes over them |
| `--device` | defaults to `cuda`; `cpu` works for smoke tests |

Sections: **A** routing sharpness per iteration · **A2** branch magnitudes ·
**B** iteration stability · **C** τ / z-score sharpening sweep · **D** rank-r
cheap-router agreement · **F** per-task routing profiles · **G** `--ablate`
paired-NLL ablation (zero the branch, force uniform routing, top-4/2/1).

Outputs into `experiments/boltzmann-moe/results/router_analysis/`:
- `routing_measure__<run>.json` — all sections. Top-level keys: `run`,
  `n_prompts`, `per_iter`, `stability`, `branches`, `sharpening`, `proxy`,
  `tasks`, `ablation`.
- `router_fit_cache__<run>.pt` — `{x, E, iter, K, tau, e_sign}`; **this is the
  frozen-backbone router-fit dataset.** `x` is fp16, `E` fp32.

Notes:
- **Fully offline.** Real activations come from the BBH few-shot prompt JSONLs
  already stored next to each checkpoint
  (`unsharded/bbh_samples_*/samples_bbh_fewshot_*.jsonl`, 27 files × 250
  samples). No dataset download, no `datasets` dependency.
- **Handles the non-MoE Hopfield baseline** — pass
  `--run math_fet_hopfield_mean_8gpt_1egpt6x_d1536_int8k_lra32_itd3_lr1p5e3_33b_16gpu`
  and it runs sections A2 and G only, skipping the routing-specific ones.
- **It does NOT yet support the legacy `BoltzmannMoE_Energy_MLP` class**, i.e.
  none of the `h1_*` / `b*` / `scale_*` checkpoints. Adding support needs an
  `E_k` branch in `Recorder._post` using the legacy formula
  `E = einsum("...h,...eh->...e", x, term1) * self._routing_scale`
  (`mlp.py:672`) instead of `expert.energy_per_token(x)`, plus reading
  `W1`/`W2` rather than a single `W`. **This is the first thing to build if you
  want to evaluate the cheap router on the W1W2 lineage** — which per §3c is
  where it pays off most.

---

## 9. What to do next

### 9a. Decisions taken this session

| Question | Decision |
|---|---|
| Target inference regime | **Prefill / batched** (compute-bound), *not* single-stream decode |
| Finetune budget | **Router-only / frozen backbone** — fit the proxy on cached activations, no training run |
| If routing is uniform | **Fix sharpness first, then top-k** |
| Scope | MoE only, **plus** the free architecture-specific win (route once, reuse across iterations) |

**Why not decode:** at batch 1 the model reads 800 MB of weights per token
against 1.2 GFLOP of math — roughly 200× bandwidth-bound. Top-k on the MoE
saves reading at most ~22 MB of 800 MB (2.8%), and the energy block's `W` is
read once and reused 6× so the iterations are nearly free on bandwidth. **FLOP
reduction in the MoE buys essentially nothing at decode.** If decode ever
becomes the target, attack the 6× iteration count or quantization instead.

**Tension to be aware of:** "router-only / frozen backbone" and "fix sharpness
first" are in conflict *if* routing is genuinely uniform. Sharpening changes the
function the model computes — going from `p ≡ 1/K` to top-1 replaces a mean of
experts with a single expert, which a frozen backbone cannot absorb. In that
branch a finetune is unavoidable; **ask before launching one.** What *is*
router-only-compatible even under uniform routing: collapsing the 8 sliced
GEMMs into 1 fused GEMM (§7.5), and deleting the branch entirely.

### 9b. Concrete queue

1. **Read job 1559723's results** and confirm §7.4/§7.5 at N=810. If confirmed,
   the honest recommendation for this checkpoint is *delete the MoE branch*
   (−25.0% FLOPs, zero cost) and report the dead-branch finding.
2. **Fix the Hopfield energy scale** so the FF branch is load-bearing at all.
   The `(1/d_int)` MEAN normalization is over-suppressing by ~3 orders of
   magnitude at these widths. Candidates: `1/√d_int` instead of `1/d_int`; a
   learnable per-layer energy gain; or raising the `scale_ff` init. This is
   upstream of everything else — a dead branch makes every routing question
   moot.
3. **Make routing scale-free**: z-scored logits **× a learned gain**, or
   `τ ≈ std_k(E_k)`, or a learnable `τ`. Direct analogue of the `1/√expert_I`
   fix that rescued the B-series. §7.4 has the sweep.
4. **Port `measure_moe_routing_20260912.py` to the legacy class** and run it on
   `h1_boltz_moe_fullsize_d768` (avg 0.501, W1W2, K=4×2048, `repulsion_coef=0.1`,
   local unsharded weights available). This is the checkpoint where (a) the FF
   branch is plausibly alive, (b) routing demonstrably specialized
   (`PROGRESS.md:40-44`, `n_dominant_experts` 14-16, `max_expert_load` 0.37),
   and (c) the router genuinely is a second matmul so the cheap proxy saves ~2×
   more.
5. **top-k WITH renormalization** A/B on an existing checkpoint. Both
   implementations do `p = p * mask` with no renorm; part of the +6.4 PPL
   post-hoc truncation cost (§5) is probably just the magnitude collapse. This
   is a one-line change and a cheap eval.
6. **Fit the rank-r + tiny-nonlinear-head router** against
   `router_fit_cache__<run>.pt`, frozen backbone, and report top-k agreement +
   end-task deltas.
7. **Re-measure iteration stability** once routing is sharp (§7.7).

### 9c. Open direction — the denominator problem

The −22% whole-model ceiling (§7.2) is **the wrong denominator for a method
claim.** It is dominated by the GPT prefix (37.5%) and the LM head (25.5%),
neither of which the routing scheme touches. Two better framings:

**(a) Block-level, iso-params / iso-active-params.** Compare a Boltz-MoE block
against a standard learned-router top-k MoE block directly, at matched total
and matched active parameters. The EGPT-block column in §7.2 (down to 0.407×)
is the number that belongs in a paper. Note the counterargument that must be
pre-empted: in a standard setting you would not spend 8 layers on a GPT
prefix — all layers would be distinct MoE layers (possibly with some mid-stack
recurrence for latent reasoning). `BOLTZ_MOE_BEST.md:83-105` already has the
matched-structure Switch ablations and shows the honest margin is only
~0.5-0.7pp avg / ~0.5 PPL.

**(b) All-Boltz-MoE, single recurrent layer.** To be faithful to the
energy-based picture, put **all** parameters in one block and recurse it — no
GPT encoding layers. Sparse MoE should scale better here than a dense EGPT,
which would need to be very wide (and therefore very FLOP-heavy) to match
params. This is the comparison that would actually establish the method. It
needs FLOPs/throughput benchmarking *before* committing to training.

**Also open — decoupling `head_dim` from `d / num_heads`.** Wide attention
(e.g. `d=2048`, `d_head=128`, `num_heads=64`, so `num_heads > d/d_head`) is
**not currently supported**: `EnergyAttention_QK` hardcodes
`self.head_dim = divide_if_divisible(hidden_size, num_heads)`
(`lm_engine/hf_models/modeling_utils/sequence_mixer_blocks/energy_attention.py:75-79`),
which forces `head_dim = d / num_heads`. Supporting `num_heads · head_dim > d`
requires `c_attn` to project to `2 · num_heads · head_dim` (it currently
projects to `2 · hidden_size`, `energy_attention.py:91-96`) and a matching
change to the output projection, which reuses the Q weights. Worth scoping as a
prerequisite for wide-attention experiments.

**Reference point for "when do people use MoE":** the existing comparison set
in `BOLTZ_MOE_BEST.md` runs Switch/Boltzmann MoE ablations down at 585-962M
total params, so this project already operates well below typical production MoE
scale. Treat any "nobody uses MoE this small" objection as a framing issue for
the paper, not a blocker for the ablations.

---

## 10. Known stale references in the existing docs

Fix these when you next touch the docs; they will send you to the wrong code.

| Doc | Claim | Reality |
|---|---|---|
| `BOLTZ_MOE_BEST.md:223` | `BoltzmannMoE_Energy_MLP` at `mlp.py:282` | it is at **`mlp.py:540`**. Line 244 is now `BoltzmannMoE_Hopfield_Energy_MLP` |
| `BOLTZ_MOE_BEST.md:225` | dispatch at `mlp_blocks/__init__.py:78` | `BoltzmannMoE_Energy_MLP` dispatch is at **`__init__.py:113`**; line 78 is now `Mixed_Energy_MLP` |
| `CLAUDE.md:37-47` "Key source files" | lists only `mlp.py` / `_BoltzmannMoEEnergyMLPArgs` | **stale w.r.t. the 2026-06-28 composable refactor.** Missing `energy_ff.py` entirely (the `FFEnergyBase` family, `build_boltzmann_moe`), and `config/mlp.py:245 _EnergyFFBoltzmannMoEArgs`. A reader following this table will not find the code that the `math_fet_boltz_hopfield*` checkpoints actually run |
| `CLAUDE.md:200-215` routing metrics | says the collapse metrics are "Logged every 10 steps" | true for the **legacy** classes only. For any `EnergyFF_*` run they were never logged until the §7.8 fix |
| `configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml:12` | "diagnosed as experts collapsing to uniform routing because (a) per-expert width 1024 is 4× narrower…" | routing *is* uniform (§7.4), but the mechanism is `E_k ≪ τ`, not expert width. And the claim was inferred — the metrics that would have shown it were not being logged |
| `SCALE_UP_PLAN.md:31-32` | notes the `h1_boltz_moe_580m` YAML header says "~580M" but the real count is 679M | correct as written — keep the warning, the config filename is still misleading |

---

## 11. SESSION FINDINGS — 2026-09-15 (routing sign, balance, Sinkhorn, energy stability)

**This session found a bug that changes how three paper claims should be read, and a
principled replacement for the property the bug was accidentally providing.** Detail
docs: `ROUTING_SIGN_BUG_20260915.md` (sign + chemical potential + Sinkhorn + energy),
`PLACEMENT_ARTIFACT_20260915.md`, `ACCEL_FINDINGS_20260915.md`,
`AVG11_ICLR_MIGRATION.md`.

### 11.1 The composable Boltzmann router is SIGN-INVERTED  ← headline

Measured on a trained checkpoint (`iclr_flops/iclr_hop_K32_top2`), reproducing the
deployed routing exactly (zscore, `e_sign="neg"`, tau=0.35, top-2):

    overlap ||gelu(W_k x)||^2/I_e of the SELECTED experts : 1.1358
    overlap averaged over ALL experts                     : 1.8457
    overlap of the LOWEST-overlap 2 experts               : 1.1358   <- identical

**The router selects the worst-matching experts, every token** (0.62x the average).
`_HopfieldExpert` stores `E = mean(gelu(Wx)^2)`, which GROWS with overlap, and
`e_sign="neg"` then picks the smallest. The legacy `BoltzmannMoE_Energy_MLP` this was
meant to reproduce does it correctly (`E = +overlap`, `softmax(+E/tau)`); the
composable refactor added a minus to the energy AND kept the legacy softmax sign.

**Scope:** every `EnergyFF_BoltzmannMoE` run — all 22 `iclr_*` arms and the live 400M.
NOT the learned-gate/Switch arms, NOT gptswitch, NOT the legacy `h1_*`/`b*`/`c*`.

Fix is opt-in: `e_sign_override: "pos"`. Verified in isolation —
overlap(chosen)/overlap(avg) is 0.68 with the default and 1.42 with the override.

### 11.2 Anti-routing was an accidental LOAD BALANCER

Correcting the sign revives the FF branch 20-100x (`ffwd/output_norm` 0.03 -> 12) and
**collapses routing** (effK 8.5 -> 1.9 of 32). The feedback sign flips:

  * anti-routing is self-LIMITING — use the worst-matching expert, it improves,
    you stop using it. Negative feedback, so load spreads on its own.
  * correct routing is self-REINFORCING — the winner gains overlap and wins more.
    The standard MoE collapse that load-balancing losses exist to prevent.

**So two paper claims are affected, and a third was measured under the bug:**
 1. "the energy-FF branch is essentially dead" (HANDOFF 7.5) — **CAUSED by the bug**.
 2. "Boltzmann routing does not collapse and needs no load-balancing loss" —
    **depends on the bug**. Also independently shaky: the deployed arm's own balance
    DEGRADES over training (see 11.5).
 3. "energy routing is at parity with a learned gate" — measured under the inverted
    sign, so the correctly-signed router is untested.

**Do not restate the routing-health results until settled.**

### 11.3 The principled replacement: load balancing is a CHEMICAL POTENTIAL

`p_k ∝ exp(E_k/tau)` is exactly the solution of `max_p sum_k p_k E_k + tau H(p)` —
the softmax IS the entropy-regularised argmax. Imposing balance as a CONSTRAINT on
the batch-marginal occupancy (`sum_tokens p_k ~ N/K`) rather than as a penalty gives

    p_k ∝ exp( (E_k - mu_k) / tau )

with `mu_k` the Lagrange multiplier — a **chemical potential**, the quantity conjugate
to occupancy. A dual variable, not a loss term.

Why this frame is worth having: it reuses the paper's own vocabulary (Boltzmann
weights, partition function, free energy), it has **no gradient pathway and no learned
gate** so the "no auxiliary loss" property survives, and it separates the two roles the
bug had collapsed together — **`E_k` decides which expert FITS, `mu_k` decides how
CROWDED it is.** Anti-routing is a fixed, crude stand-in for `mu_k`: "prefer the expert
you match least" is a static proxy for "prefer the under-occupied expert". Correlated,
hence the balance, but it pays by inverting the SELECTION.

### 11.4 tau x balance sweep at the corrected sign (400M shape, 300 steps)

    arm  tau  mu_k | effK/32  ffwd_norm      lm_loss
    S1  0.35   --  |   4.35     0.582        5.9383   deployed (anti-routing)
    S2  0.35  off  |   2.32    12.44 (121x)  5.9443
    T2  1.0   off  |   4.47    12.13 (129x)  5.9288
    T3  3.0   off  |   7.75     2.36 ( 21x)  5.9464
    T4  0.35   ON  |   3.69    13.94 (160x)  5.9674
    T5  1.0    ON  |   7.45     9.75 (122x)  5.9177  <- best overall

**tau and mu_k are COMPLEMENTARY, not redundant.** mu_k helps at each tau
(2.32->3.69 at 0.35; 4.47->7.45 at 1.0) and tau helps at each mu
(2.32->4.47->7.75 off; 3.69->7.45 on). **tau ALONE is self-defeating**: T3 buys
balance but leaves the branch at 2.36, 5-6x below every mu_k arm, because masked
top-k weights shrink as routing softens.

⚠ **A step-30 reading of this sweep said "tau is not needed" and was WRONG** — step-200
data refuted it. S2 had already shown non-monotone early dynamics (1.03 -> 1.47 ->
2.32). **Do not draw conclusions from this system before ~200 steps.**

### 11.5 SINKHORN — the exact dual beats the clamped control rule decisively  ← ship this

`balance_rate` reaches balance by proportional control on a per-expert bias, and it sat
**PINNED at the +-1.0 `_BIAS_MAX` clamp in every arm** — delivering its result while
saturated. Raising the clamp is the obvious and wrong move (the bound exists because an
unclamped `sign()`-based version hit |bias| = 1482 and destabilised an arm, loss
4.05 -> 5.18 with 77 upward jumps).

`sinkhorn_iters` removes the multiplier: solve the dual exactly by log-domain
iteration `mu <- mu + log(load(mu) * K)`. No clamp, no gain to tune, exact not lagged.
Solved under `no_grad`, so `mu` is constant w.r.t. differentiation — correct for a
Lagrange multiplier, and it keeps the no-gradient-pathway property.

**134M results (built from `iclr_hop_K32_top2`, the BEST 134M Boltzmann arm at
Avg11 44.38; 2000-step budget, matched step 340, uniform max-share would be 0.031):**

    arm                          lm_loss  effK/32  max_share  E_mean  E_max     mu    cos
    M1 shipped (INVERTED)         5.0858     5.04     0.3533  0.5507  14.19     --  0.1226
    M2 corrected + clamped bias   5.1053    12.10     0.1774  0.1809   4.63  1.000  0.0598
    M3 corrected + SINKHORN       5.0392    26.57     0.0718  0.2157  14.00  3.440  0.0582

  * **Sinkhorn is 5.3x better balanced than the shipped arm** (effK 26.6 vs 5.0) and
    2.2x better than the clamped bias. `max_share` 0.072 against the 0.031 ideal,
    where the SHIPPED arm has one expert taking **35% of all tokens**.
  * It gets there because **mu reaches 3.44 — 3.4x past the clamp** M2 is pinned at.
    The clamp WAS the binding constraint, exactly as predicted.
  * M3 also has the **best loss** (5.0392 vs M1's 5.0858) and the **best expert
    diversity** (cos 0.058 vs 0.123). It is not buying balance by homogenising experts.
  * Loss gap 0.047 sits near a 0.038 noise floor (measured at a different shape), so
    treat it as suggestive, not established.

**effK TRENDS matter as much as the levels:**

    M1 shipped   13.8 -> 19.9 -> 12.2 -> 5.2 -> 4.7 -> 4.9 -> 5.4 -> 5.7   PEAKED then COLLAPSED
    M2 clamped    8.3 ->  7.8 ->  8.9 -> 9.4 -> 10.3 -> 11.1 -> 12.1 -> 13.3  steadily improving
    M3 sinkhorn  31.6 ->  9.4 -> 14.3 -> 16.6 -> 20.9 -> 24.6 -> 25.8 -> 26.6  steadily improving

**The shipped arm's balance DEGRADES over training while both balanced corrected arms
IMPROVE.** That is independent evidence against "Boltzmann routing does not collapse".

Sinkhorn engineering checks: converges in **3 iterations** (mu 0.165 -> 0.179 -> 0.180,
then flat); **1 graph, 0 graph breaks** under `torch.compile`, no recompilation;
survives activation checkpointing across {0,3,10} iters x {no,with} repulsion; and
**verified a 0.000e+00 no-op by default** against a clean HEAD worktree with identical
state_dict keys — twice, because the live 400M arm reads this code.

Two documented approximations: **TRAIN-ONLY** (load is a batch property, so applying it
at inference would make routing depend on batch composition — the standard
Sinkhorn-router choice, and it does introduce a train/inference mismatch), and the load
is the **LOCAL per-rank batch** (a global constraint needs a collective, deliberately
avoided given the multi-node hang in 11.7).

### 11.6 ENERGY STABILITY — the feared runaway does not happen, and the SHIPPED arm is the worst

Concern: the block ASCENDS the energy (`out = +grad E`) and the corrected router now
selects the HIGHEST-energy experts — positive feedback that could explode late.

**Structural answer:** the energy is evaluated on `ln_x = self.ln(x)` (RMSNorm), NOT the
raw residual (`models/energy/layer.py:857,1026`). So `E` cannot run away through the
residual growing; the only path left is `||W||`, in which `E` is quadratic. A SOFT
bound — RMSNorm has a learnable gain and only `weight_decay: 0.1` opposes `||W||`.

**Empirical answer (`energy_abs_mean`, added this session):**

    M1 shipped INVERTED  0.107 0.157 0.364 0.547 0.590 0.533 0.504 0.476 0.448  grew 5x, peaked, declining
    M2 corrected+clamped 0.106 0.135 0.250 0.200 0.180 0.162 0.172 0.164 0.186  peaked early, stable
    M3 corrected+sinkhorn 0.108 0.150 0.149 0.158 0.213 0.222 0.217 0.216       PLATEAUED

**The corrected arms carry 2.5-3x SMALLER energy than what is currently shipped.** The
inverted sign is the one that grew 5x. Mechanistically sensible: anti-routing selects
the LOWEST-energy experts and then ascends them, systematically pushing the bottom of
the distribution up. `energy_abs_max` is comparable (M3 14.0 vs M1 14.2), so Sinkhorn is
not introducing worse outliers than production already has.

**Conclusion: no activation change is warranted.** If `E` ever does trend up, the ranked
fix is (1) **weight-normalise the energy**, `E_k = mean(gelu(W_k x / ||W_k||)^2)` — one
line, keeps GELU and the landscape, makes `E` scale-invariant in `W`, and **retires the
`routing_norm: zscore` patch**, which exists for the same arbitrary-scale problem seen
from the too-SMALL side; (2) **normalised descent step / trust region**, leaving the
energy untouched and bounding only the step (`pref` is already just a fixed step size);
(3) **bounded phi (sigmoid/tanh) LAST** — changing phi has a hard negative result here
(`tanh_exact` lost 2.2pp avg / +3.4 PPL) and saturation FLATTENS the energy across
experts, recreating the uniform-routing failure from the opposite direction. Bounding by
saturation costs routing signal; bounding by normalisation does not.

### 11.7 Two invalidated performance conclusions

**(a) The "~5x slower than gptswitch" figure is substantially HOST PLACEMENT.** Same
unfused code, same config, resumed from the same checkpoint on a different host pair:
**6.72-6.81 -> 2.45-2.53 s/step**, 2.7x from placement alone. Identical GPU
model/driver/`gpu_factor`, no MIG; sibling contention ruled out (median 6.739 before
gptswitch finished vs 6.784 after, n=611/89). Likely dataloader starvation from shared
CPU slots: per-rank GPU-busy is 1.71 s, so 25% utilisation at 6.76 s wall vs 68% at
2.49 s. Real ratio ~1.9x, and even that is not placement-controlled.
**Rule: no multi-node s/step claim is admissible unless placement-controlled.**

**(b) The "launch-overhead bound / 79.5k kernels" reading is largely void** — it rested
on 1.71 s GPU-busy against 6.75 s wall (25% util). At 68% the step is much closer to
compute-bound.

**(c) `fused_experts` is validated SINGLE-NODE ONLY.** EXACT (float64 1.227e-15) and
1.61x at 4 GPU / 1 node, but it **WEDGED at 16 GPU / 2 nodes** — 17+ min with ZERO
inductor cache writes against 98 s to first step unfused. Reverted on the live arm.
Process lesson: exactness tests, bf16 bit-identity, checkpointing tests and an A'
replicate ALL passed — none of them can see compilation or collectives. **Exactness
does not transfer across parallelism shapes.** Bisect recorded in ACCEL_FINDINGS.

### 11.8 Repulsion: cost corrected, and it IS load-bearing

  * **Repulsion is 17-21% of the optimizer step, NOT 2-4%.** The old 2-4% came from a
    UNITS error: it divided 48 block-calls by 1536 per-expert projections, but the
    bench's 7.59 ms marginal was measured PER BLOCK CALL and already includes all 32
    experts. Correct: 7.59 x 48 = ~364 ms of 1713 ms = 21%, matched by a direct A/B
    (1.463 vs 1.210 s/step = 17%).
  * **It is load-bearing**: with NO repulsion, expert output alignment sits at 0.43 and
    RISES to 0.51 — it never decays. Every repulsion arm decays instead. So the decay
    is CAUSED by the force, not by experts settling into niches.
  * **But the benefit is steeply front-loaded**: 0 -> 0.5 pair-applications/call buys
    4.7x better alignment for 2.5% of the step; 0.5 -> 4.0 buys a further 4.4x for 15%.
  * **Intermittent firing weakens the regulariser** (alignment plateaus 4-6x higher), so
    prefer **`repulsion_space: "weight"`** — 2.20 vs 7.59 ms/call, N-INDEPENDENT
    (2.21/2.20/2.33 at N=2048/4096/8192 vs 4.06/7.59/14.91), sparse-kernel compatible,
    and full strength every step. Coefficient does NOT transfer: weight cosines are
    ~5-25x smaller than output cosines, so re-sweep (gradient-matched estimate ~0.7,
    aux-matched ~4.0; bracket {0.5, 2, 8}).

### 11.9 Cheap router: the learnable proxy works, the spectral one does not

  * **Learnable rank-8 proxy, distilled online: 0.942 top-2 agreement** with the exact
    router (chance 0.0625), matching the offline fitted-head study's 0.94 at r=8.
  * **The spectral `||Wx||^2` proxy FAILS, and ReLU does not rescue it.** Top-2
    agreement 0.214 (GELU) vs 0.209 (ReLU) — ReLU marginally WORSE; rank-r spectral is
    at or BELOW chance. Reason: the positive-part FRACTION varies per expert per token
    and carries the discriminative signal, which `sum_i z_i^2` discards. So 7.6's
    diagnosis of "the nonlinearity" named the wrong culprit.
  * **Note a conflation in older notes:** 7.6's WORKING r=8 result kept the nonlinearity
    INSIDE the rank-r subspace (`mean(gelu(W^(r)x)^2)`), which is a different and
    costlier construction than the spectral `||W^(r)x||^2`. Only the first works.
  * A **two-moment** proxy is the cheap winner in testing: 0.895 top-2 at `d(1+r)`
    (~57x cheaper than exact), beating the L1 form that costs 512x more. `mu` alone —
    one dot product, cost `d` — already gets 0.638.
  * **True sparsity is NOT implemented.** `top_k` is a post-hoc MASK: all 32 experts'
    forward AND back projections are computed then multiplied by a `p` that is zero for
    30 of them. Back-projection is the free half (~13-15% of the step, no router
    needed); the forward half needs the proxy and cuts the 61% elementwise bucket ~16x.

### 11.10 What to do next  ⚠ SUPERSEDED by §12.7 — and several §11 numbers predate the mu-at-eval fix of §12.1

 1. **Let the 134M arms finish (2000 steps)** and confirm M3 > M1 on loss and that
    `energy_abs_mean` stays plateaued.
 2. **Then a 400M Sinkhorn arm** — but validate MULTI-NODE on the throwaway
    `configs/iclr_scale/scale32B_boltz_hop_PROFILE.yml` FIRST. Sinkhorn's CPU dynamo
    check is clean (1 graph, 0 breaks) but that is not inductor+FSDP, and this is
    exactly the step that was skipped before `fused_experts` wedged the live arm.
 3. **Paper**: hold all three routing-health claims (11.2). The chemical-potential
    derivation is a STRONGER replacement for claim 2, not a retraction — balance as the
    dual of a capacity constraint, aux-loss-free and derivable rather than heuristic.
    None of the efficiency results are affected, nor anything about Switch/gptswitch.
 4. **Unresolved**: no quality evidence at this scale. Every 300-step arm sat inside the
    lm_loss noise floor, and 7.5 found deleting the FF branch entirely moved perplexity
    by +0.0003. "Corrected routing is better" needs a long run with a downstream eval.


---

## 12. SESSION STANDING — 2026-09-16 (READ THIS FIRST)

Section 11 is still correct on the routing sign and Sinkhorn, but **11.10's "what to do next" is
superseded** and several §11 numbers were measured before a large inference bug was found. Start
here.

### 12.1 The one thing that changed everything: mu never reached inference

Sinkhorn's dual `mu` was solved per forward CALL, applied only under `self.training`, and never
stored. So every Sinkhorn model was **trained with mu-tilted routing and EVALUATED with mu = 0**.

Measured on identical held-out web batches, train vs eval mode:

| arm | train CE | eval CE | discontinuity |
|---|---:|---:|---:|
| PURE isoP + sinkhorn | 3.4088 | 4.9959 | **+1.587** |
| PURE T12 + sinkhorn (12 iters) | 3.3403 | 5.1673 | **+1.827** |
| PURE isoP + clamped bias | 3.3360 | 3.3365 | +0.0005 |
| HYBRID K16 + sinkhorn | 2.9454 | 2.9486 | +0.0032 |

The cost scales with how much of the net depends on mu: all iterations of a pure stack, 1 block of
7 in a hybrid. The clamped `load_balance_bias` shows ~zero because it is a persistent buffer with
no `self.training` gate, so it always reached eval.

**FIXED** by `sinkhorn_persist_mu` + `sinkhorn_mu_iters` (a PER-ITERATION buffer, cycled by call
index — one shared buffer cannot work, the duals differ across iterations by 1.56-1.90 mean spread
and individual experts flip sign, e.g. -1.27 at iter 0 to +1.39 at iter 1). After the fix the
discontinuity is **-0.058**, i.e. gone. `calibrate_sinkhorn_mu_20260915.py` recovers mu for an
already-trained checkpoint in minutes without retraining.

**REQUIRED on every new Sinkhorn arm:** `sinkhorn_persist_mu: true` and `sinkhorn_mu_iters` = that
block's `layer_iterations` entry. Optional for hybrids (0.003 nats) but free.

### 12.2 The bug INVERTED the depth scaling law — the most consequential finding

bits/byte on wikitext (report THIS, not `word_perplexity`, which is `exp(bpb * 3.7)` and turns a
1.55x regression into a 7.8x one):

| arm | iters | mu DROPPED | mu RESTORED |
|---|---:|---:|---:|
| hybrid K16 (1 MoE of 7) | 6 | 1.0000 | — |
| pure big | 4 | 1.2524 | 1.2103 |
| pure 1blk | 8 | 1.3966 | 1.1525 |
| pure isoP | 8 | 1.5530 | 1.1201 |
| **pure T12** | **12** | **1.6034** | **1.0996** |

Both columns monotone, **opposite directions**. Under the bug deeper is worse; fixed, deeper is
BETTER, and T12 becomes the best pure model (Avg11 40.96, bpb 1.0996). Anyone reading the pre-fix
numbers would have concluded depth hurts, most confidently from the arm that proves the opposite.
**Any recurrence-depth claim measured before 2026-09-16 is suspect.**

### 12.3 Corrected-sign grid: 14 arms, all token-matched but one

Mean Avg11 delta over the six HYBRID arms is **+0.03pp** — the sign correction is quality-neutral,
so the paper's parity claims survive. Best rows: K32-top2 **44.58**, renorm 44.54, K16-dense 44.40,
K32-top1 44.19, K16-top2 43.50. Pure arms (mu restored): T12 40.96, bal_corr 41.49, 1blk 40.43,
isoP 40.42.

**`iclr_big_hop_pure_sink` is NOT comparable** — registered at 4 GPUs while its published
counterpart ran at 8, so it saw 1.97B tokens against 3.93B. Its -0.59pp is undertraining, not the
sign. Strike it from any analysis. 13 of 14 match to 0.01%.

### 12.4 Two BROKEN SCHEDULES, one of them in the published configs

`slope90k_*` set `num_training_steps: 90000` but warmup 2000 + decay 28000 = **30000**, so 60000
steps ran pinned at the 2e-4 floor. Loss falls 0.12-0.15 per 10k while decaying, then
-0.002..0.000 at the floor: the last 45-51k steps bought 0.015-0.019 nats. **The bug is in the
published `iclr_slope` configs**, so the paper's "+0.41pp for 3x the tokens" is a lower bound, and
the conclusion drawn from it has been DELETED from the paper.

Hence §12.6's replacement arms. This is check 1 of the new CLAUDE.md pre-flight.

### 12.5 LR: peak matters, floor does not, and 1e-2 is on a stability boundary

Pure isoP, 5000-6000 step grids at 262144 tok/step:

* **FLOOR is inert.** 2e-4 vs 2e-5 at peak 2e-3: -0.021. 1e-3 vs 2e-5 at peak 1e-2 (a 50x range):
  +0.008. Both inside the 0.038 noise floor, and the two contrasts disagree in sign.
* **PEAK is worth ~0.06 nats.** 2e-3 -> 1e-2 gives +0.058 at floor 2e-4 and +0.069 at floor 2e-5.
* **But 1e-2 diverges ~1 in 3.** Three arms at identical peak 1e-2 and the same default seed 42:
  two healthy (grad_norm 3.1), one blew up (grad_norm **7045**, loss 6.44 -> 8.34, ending 6.39, no
  recovery under cosine decay). `gradient_clipping: 1` was ACTIVE and did not prevent it. A reseed
  at seed 7 trained cleanly to 3.8521, the best of the grid.
* 2e-2 diverges outright (7.89). Lower LRs are simply worse.

**Recommendation: keep 2e-3 for long runs.** 0.06 nats is not worth a ~1-in-3 divergence, and the
mu fix was worth 1.6-1.8 nats, ~30x more. The pure model's steeper log-log slope (-0.097 vs -0.082
hybrid) means it wants more TOKENS, not a bigger step.

### 12.6 RUNNING RIGHT NOW

| arm | queue | GPUs | progress | note |
|---|---|---|---|---|
| `scale32B_boltz_sinkhorn` | normal | **16** | 34.5k/61035 (57%) | 400M HYBRID, 32B tokens, schedule OK, all knobs right except `persist_mu` (0.003 nats, recalibratable). **Do not restart.** ~1.5 days left |
| `t90k_switch_lastisoP` | preemptable | 4 | fresh | 90k = 23.6B, correctly scheduled. ~7.4 h |
| `t90k_hybrid_K32top2` | preemptable | 4 | fresh | ~12.9 h |
| `t90k_pure_T12` | preemptable | 4 | fresh | ~37.8 h |
| `slope90k_{1blk,hyb}_sink` | preemptable | 4 each | ~75-82k/90k | BROKEN schedule; kept only for like-for-like vs the published 44.32 |
| `iclr_big_hop_sandwich_sink` | preemptable | 4 | 8k/15k | |

`grp_ebm` is **32/32**: another group member's job (16) + our 400M arm (16). Everything else is
on `grp_preemptable` (1397/6144). `blimits`, not `bjobs`, is the authoritative quota check.

### 12.7 WHAT WE ARE FOCUSED ON NEXT

1. **TRUE SPARSITY — in progress, half landed.** `sparse_backproj` (opt-in, default off) does the
   back-projection over only the top-k experts via capacity dispatch: fixed `(K, C, I_e)` buffer +
   one `bmm` + scatter-add. **Verified EXACT** vs the dense mask (relative error 3.79e-16, overflow
   0); overflow drops pairs and is COUNTED in `_sparse_overflow`. Back-GEMM saving 6.40x at K=16
   k=2, 12.80x at K=32 k=2. Deliberately not a per-expert loop — that measured 0.59x, SLOWER.
   * ~~**NOT yet measured: actual step-time delta.**~~ **MEASURED — see 12.9.** sparse_backproj
     alone is 1.18-1.35x compiled at the real batch size and a LOSS (0.44-0.94x) at a small one.
     The full `sparse_forward` is what pays: **4.69x** compiled fwd+bwd on the pure arm, past the
     "cannot beat 2x" floor. Both depend strongly on tokens/call.
   * **The big half is still blocked**: the forward projection cannot be skipped by an exact router
     (the 1/2(1+k/K) floor) and needs the proxy router (0.942 top-2 agreement at rank 8). That is
     where the 61% elementwise bucket lives, and where the pure design pays off most — the mixture
     is 98.9-99.6% of per-token FLOPs in a pure stack against ~22% in a hybrid, so sparsity is
     end-to-end there rather than capped at 22%.
2. **The 3-family x 2-scale matrix.** Have: 134M @ 7.86B complete for hybrid / GPT-switch / pure;
   400M @ 32B for GPT-switch (done) and hybrid (running). **Missing: 400M PURE** (lower priority
   per the user) and, until §12.6's arms land, any sound 134M run beyond 7.86B.
3. **Paper.** Pushed through `48da63b`. Open `\CC` items: the un-remeasured expert-cosine bound;
   the token-scaling paragraph pending §12.6; and the 400M pure router comparison, whose "+0.19pp
   in favour of energy" measures the SIGN-INVERTED router and must not be quoted as support.
4. ~~**Untracked and NOT in the repo**: `experiments/eval_scripts/compute_avg11.py` plus six Avg11
   helpers.~~ **DONE 2026-09-16** — `compute_avg11.py` and the six helpers are now tracked
   (commit `8e726fa3`), so a fresh clone can reproduce every headline number. It is THE reference
   implementation of the metric; see the banner at the top of this file.

### 12.8 Process rules added this session

`CLAUDE.md` now opens with a **10-point config pre-flight**, mandatory before any long run. Each
point comes from a bug that cost real compute: schedule coverage, tokens/step matching, hosts vs
GPUs, `load_args`, the checkpoint pointer vs `max_to_keep`, glob prefix collisions
(`foo_*` matches `foo_sink_*` and `foo_s7_*` — this produced two wrong numbers in one session),
knobs reaching the model rather than just the YAML, the per-expert-kind sign direction, the
Sinkhorn requirements, and diffing against the arm being copied.


### 12.9 TRUE SPARSITY — the other half landed, and the wall clock disagrees with the FLOPs

**What is in the code now** (`energy_ff.py`, opt-in, every default unchanged):

* `sparse_forward` — skips the forward AND back projection of the K-k experts a token was not
  routed to. Requires `proxy_rank > 0` and `fused_experts` (asserted): an exact router cannot
  skip the forward projection, because it needs all K energies to decide. That is the
  `1/2*(1+k/K)` floor, and the proxy is what breaks it.
* `proxy_route` now DOES SOMETHING. It was plumbed through config, builder and constructor and
  then **never read** — `self.proxy_route` was assigned and that was the end of it. Pre-flight
  check 7, in the file that check was written for. It now moves SELECTION to the cheap router
  while the WEIGHTS stay exact, in the dense path too, which is the controlled A/B for selection
  quality with the dispatch machinery held out.
* `sparse_backproj`, `sparse_forward` and `sparse_capacity_factor` reached the pydantic args
  class and `get_mlp_block` for the first time. Yesterday's `sparse_backproj` was reachable ONLY
  by calling `build_boltzmann_moe` directly — which is exactly what its test does, so the test
  passed while a YAML using the flag would have been REJECTED outright (`extra="forbid"`). Loud
  rather than silent, so no wrong run came of it, but the flag was unusable.
* `_logits` split into `_logits_raw` + `_mu_for` so the dual is solved, and `_mu_call` ticked,
  exactly ONCE per forward even though the sparse path needs the logit map twice.
* `_dispatch_plan` factored out and shared by both sparse paths so they cannot drift.

**Exactness.** `test_sparse_forward_20260916.py`. With an ORACLE proxy (returns the exact
energies) the sparse path reproduces the dense output to **4.3e-16 - 8.0e-16** with the selection
left FREE, in all three of `routing_norm=none/renormalize`, `zscore/renormalize` and
`zscore/masked`. So **the proxy's prediction error is the ENTIRE approximation** — there is no
second error hiding in the dispatch, the denominator completion or the zscore moments. Forcing
the dense selection also gives bit-exactness (7.98e-16, 5.05e-16), and overflow is counted, not
silent. Two smaller approximations can be removed by config rather than by code:
`renormalize_topk: true` deletes the proxy-completed denominator term entirely (and was measured
at Avg11 44.54 against 44.58 for the masked form, i.e. free), and `routing_norm: none|sqrt_width`
needs no per-token moments.

**Wall clock — H100, one GPU, ms per block call, speedup vs the dense mask.** Measured four ways
because the answer CHANGES SIGN between them. `torch_compile: true` on every arm and
forward+backward is what training pays, so **the compiled f+b column at the arm's real per-call
token count is the only admissible training number**; the rest is there to show why.

tokens/call = 4096:

| shape | eager fwd | eager f+b | compiled fwd | compiled f+b |
|---|---:|---:|---:|---:|
| pure_T12 K=16 I_e=4480 · backproj | 1.02x | 1.09x | 0.89x | 0.94x |
| pure_T12 · **sparse_forward** | 2.04x | 2.82x | 2.14x | 1.36x |
| hyb_K32 K=32 I_e=512 · backproj | 0.77x | 0.89x | 0.44x | 0.53x |
| hyb_K32 · **sparse_forward** | 0.72x | 0.80x | 0.39x | 0.41x |
| big_hyb K=16 I_e=1280 · backproj | 0.79x | 0.95x | 0.57x | 0.66x |
| big_hyb · **sparse_forward** | 0.91x | 0.99x | 0.62x | 0.49x |

tokens/call = 16384 (dense f+b compiled: 24.67 / 5.44 / 8.46 ms):

| shape | eager f+b | compiled fwd | **compiled f+b** |
|---|---:|---:|---:|
| pure_T12 · backproj | 1.17x | 1.29x | 1.35x |
| pure_T12 · **sparse_forward** | 4.96x | 4.33x | **4.69x** |
| hyb_K32 · backproj | 1.12x | 0.98x | 1.18x |
| hyb_K32 · **sparse_forward** | 2.87x | 1.50x | **2.09x** |
| big_hyb · backproj | 1.08x | 1.02x | 1.21x |
| big_hyb · **sparse_forward** | 2.54x | 1.95x | **2.37x** |

Four things follow, two of which correct claims of mine:

1. **The saving is strongly batch-size dependent and the sign flips.** At 4096 tokens/call
   `sparse_forward` LOSES on both hybrid shapes (0.41x, 0.49x compiled); at 16384 it wins
   2.1-2.4x. The dispatch overhead is O(T*k) index work independent of `I_e`, while the saving
   is O(T*(K-k)*I_e), so it needs a wide expert AND enough tokens for the batched GEMM to be
   efficient — C = cf*T*k/K is 640 at T=4096 against 2560 at T=16384. **Any sparsity claim must
   state the per-call token count**; a single number is meaningless.
2. **`sparse_backproj` is small but not nothing: 1.18-1.35x compiled at 16384.** §12.7 predicted
   ~12% from FLOP arithmetic; that was right by accident at the wrong batch size and wrong at
   4096 (0.44-0.94x, i.e. a LOSS). It is dominated by `sparse_forward` everywhere and is now
   redundant with it, so it is not worth shipping on its own — but "worthless" overstated it.
3. **4.69x on the pure arm is past the "cannot beat 2x" floor, measured.** That floor applies to
   an exact router, which must compute all K energies before it can choose. The proxy is what
   breaks it, and this is the number that says the idea works rather than merely type-checks.
   4.96x eager against a 6.36x FLOP bound is 78% of theoretical, so the residue is dispatch
   overhead, not something structural.
4. **The 400M hybrid is the worst case exactly as configured.** `micro_batch_size: 1` x 4096 =
   4096 tokens/call — the column where sparsity loses (0.49x compiled). Raising micro_batch_size
   and cutting `gradient_accumulation_steps` to match keeps tokens/step identical and moves it
   into the winning regime. Cheap, and UNTESTED. Do not enable sparsity on that arm without it.

Net: this is a **pure-model lever**, which is where §12.7 expected it — the mixture is 98.9-99.6%
of per-token FLOPs in a pure stack against ~22% in a hybrid, and the pure design also wants the
wide experts that sparsity needs. It also cuts the intermediate activation footprint by
cf*k/K, i.e. 6.4x at K=16 k=2 (0.55 GiB -> 0.09 GiB at T=4096, bf16), which is what limits `I_e`.

**Not yet measured: the quality cost.** `calibrate_proxy_router_20260916.py` fits the proxy on a
trained checkpoint with no retraining — `proxy_V` from the SVD of each trained `W_k` (sound
because HANDOFF 7.3 measured the trained expert weights near rank-2), then least squares for the
2r+1 head coefficients, then it reports genuine per-row top-k set agreement and, on a recurrent
stack, the PER-ITERATION breakdown. That last column is the risk: one proxy serves all 12
applications of a shared block, which is the structure that made a single shared `sinkhorn_mu`
buffer fail (§12.1). Written and parses; NOT yet run against a checkpoint.

### 12.10 t90k_pure_T12 restarted from step 0, twice, and the mechanism is general

Found while checking arm health: T12 was at step 2670, then at step 1030 with
`learning_rate = 1.03e-3` — warmup, i.e. a fresh run. Two `step = 10` lines in its log. ~2670
steps lost.

**Mechanism.** The watchdog resolves `load_args` at SUBMIT time. LSF preemption REQUEUES a job on
the same jid and re-runs the ORIGINAL command, so an arm launched *before its first checkpoint
existed* had no `load_args` baked in and restarted from step 0 on every preemption, indefinitely.
At 1.59 s/step over 90000 steps this arm needs ~40 h uninterrupted on `preemptable`; it would
never have finished. `switch` and `hybrid` were unaffected only because they have not been
preempted yet — they were equally exposed.

**Fixed two ways.** (a) `load_args` appended to all three `configs/tok90k/*.yml`, verified through
the real loader to point at a checkpoint dir that exists (latest 22000 / 11000 / 1000) — this
protects the jobs already submitted, whose baked-in command reads the base config. Duplicate
`load_args` from the watchdog's own append is harmless: PyYAML takes last-wins and both name the
same path. (b) `watchdog_loop.sh` now resolves `load_args` INSIDE the job script, so every
requeue re-evaluates it; patched by atomic rename so the running watchdog's bash is not disturbed.
It logs `RESUME:` lines, which is the thing to grep for after any preemption.

One oddity left alone: `t90k_pure_T12/global_step2000` is a stale checkpoint from the first
attempt, newer in step number than the pointer (1000) but from a different run. The trainer only
ever reads the pointer, so it is inert — do not "fix" it by pointing at 2000.


### 12.11 The proxy router: the quad head FAILS, and §7.6's 0.90 was never a head measurement

**The negative result.** Post-hoc fit of the shipped diagonal-quadratic proxy on
`pure_hop_T12_sink` (K=16, top-2, 98,304 tokens from the training corpus tail). Chance floor for
top-2 of 16 is `k/K` = **0.125**, verified empirically at 0.1256:

| r | top-2 agreement | energy R² | x chance |
|---:|---:|---:|---:|
| 2 | 0.2460 | 0.324 | 2.0x |
| 8 | **0.3481** | 0.572 | 2.8x |
| 16 | 0.4101 | 0.682 | 3.3x |
| 32 | 0.4614 | 0.760 | 3.7x |

Above chance, nowhere near usable: routing on this would change which experts fire for ~2/3 of
tokens. **§12.9's "0.942 top-2 agreement at rank 8" is wrong** and is corrected there.

**Cause 1: a documentation error that propagated.** §7.6 measured the EXACT energy restricted to
a rank-r subspace and got 0.94/0.90; it then RECOMMENDED a fitted head as the implementation. The
0.90 was carried forward as though it described the head — the code comment records the slide
verbatim ("What works is the exact energy restricted to a rank-r subspace: r=8 gave 0.94... This
is the trainable version: a learned per-expert projection V_k plus a diagonal-quadratic head").
§7.6 now carries a correction box.

**Cause 2: one proxy cannot serve a recurrent stack.** Grouping the 96 block calls by position in
the 12-iteration cycle (means over 8 batches):

| cycle pos | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **11** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| agreement | .436 | .202 | **.080** | .251 | .229 | .278 | .344 | .317 | .417 | .482 | .551 | **.592** |

A **7.4x spread**, and at position 2 it is **below the 0.125 chance floor** — anti-correlated with
the true router. Same structure that defeated a single shared `sinkhorn_mu` buffer (§12.1): a
shared block solves a different problem at each application.

**Cause 3: the wrong objective.** The fit was closed-form least squares on energy **MSE**, while
the metric is top-k SET agreement. LS spends capacity on the magnitude of experts that will never
be selected and gets no credit for ordering the two that will.

**What landed in response** (`proxy_kind`, `proxy_out_dim`, `proxy_iters`; all default to the old
behaviour):

* `proxy_kind: "subspace"` — `E_hat_k = mean(gelu(B_k a_k)²)·s_k + b_k` with `B_k = W_k V_k`.
  `W_k P_k x = B_k a_k` exactly, so the ONLY error is rank truncation, not function class.
* **`proxy_out_dim` = m, and it is not optional.** `B_k` is `(I_e, r)`, so the naive subspace form
  materialises `(T, K, I_e)` — **the same elementwise work and the same activation footprint as the
  DENSE path**, cancelling two of sparsity's three savings and leaving only the GEMM. `E_k` is a
  MEAN over `I_e` coordinates, so an m-row subsample is an UNBIASED estimator with variance ~1/m.
  At K=16, r=8, d=768, I_e=4480: m=all is 672K MAC but 71,680 elementwise (no better than dense);
  **m=512 is 164K MAC and 8,192 elementwise** — 0.15% of dense MACs and 11% of its elementwise.
* `proxy_iters` — one head per iteration, cycled by call index. **EVAL/CALIBRATION ONLY, asserted.**
  A call counter is unsound in training under activation checkpointing, which replays the forward
  during backward; `gradient_checkpointing_method: block` is set on the 400M arm. Training with
  per-iteration heads requires the block to be told its iteration index, which it is not.
* `fit_subspace_proxy_20260916.py` — data-aware basis (SVD of `W_k Σ^{1/2}`, i.e. the best rank-r
  approximation in the DATA metric rather than of `W_k` alone), closed-form scale/bias, then
  **gradient refinement on the routing KL**. All agreement numbers on a HELD-OUT split, since the
  refinement fits thousands of parameters.

**Consequence for §12.9's speedup, stated plainly:** the exact mechanism (`sparse_backproj`) is
capped at **1.73x** by `1/2(1+cf·k/K)` and delivers 1.18-1.35x. The 4.69x requires
`sparse_forward`, which requires a working selector. Until agreement is high, **4.69x is a
capability, not a result.**

### 12.12 Why the retrofit fails: with `renormalize_topk: false`, top-k is a SCALING, not a sparsification

This is the structural fact behind §12.11's numbers, and it decides how a sparse arm must be
configured.

`_route` computes `p = softmax(all K logits)` and then ZEROES all but the top-k, leaving
`sum(p) < 1`. So the surviving weights depend on **every** expert's energy through the shared
denominator. The top-k mask does not remove the other experts from the computation; it removes
their OUTPUT while keeping their influence on the SCALE of the ones that remain.

How big is that influence? From `t90k_pure_T12`'s own training log:

```
load_mean_token_entropy   = 0.523-0.536   (normalized, so effective experts/token = K^H)
=> K^H = 16^0.536 = 4.42 of 16
=> a top-2 mask captures ~2/4.42 = 45% of the softmax mass
=> the UNCOMPUTED tail is ~55% of the denominator
```

which matches the independently-known `sum(p) ~= 0.45` for this family. **A sparse path that never
evaluates K-k experts is therefore missing ~55% of the quantity that sets its own output
magnitude**, and it must either estimate it or not need it. That is why:

* over-selection barely helps (p=8 still leaves ~8 miscalibrated terms: bpb 3.03 against a dense
  1.0996), and why the curve is DISCONTINUOUS at p=K, where the tail vanishes and the zscore
  moments become exact in the same step: 3.03 -> 1.0996;
* `renormalize_topk: true` fixes it structurally -- there is no all-K sum to estimate;
* and a proxy trained on ranking alone cannot supply it. KL is nearly invariant to the energies'
  absolute scale, so `--mse_coef` adds a calibration term. Whether that is enough to rescue the
  RETROFIT is what job 1706755 measures; it is not needed at all for an arm TRAINED with
  renormalisation.

**Consequence for the retrain (TODO gate 2):** set `renormalize_topk: true`. Retrofitting it costs
+0.483 bpb because `scale_ff` was trained against `sum(p) ~= 0.45`, but TRAINING with it is free --
§12.3 measured `iclr_hop_K16_top2_renorm` at Avg11 **44.54** against **44.58** for the masked form.
With the all-K denominator gone, the remaining approximations are the zscore moments (+0.318
retrofitted, removable with `routing_norm: sqrt_width`) and SELECTION, at **+0.0165**.

A note for anyone tempted by the masked form's rationale: leaving `sum(p) < 1` was chosen to avoid
abrupt weight redistribution at routing boundaries (see the comment in `_route`). That choice is
what makes the design un-sparsifiable, and the arms trained with renormalisation show it costs
nothing to give up.

### 12.13 Sparsity: FINAL numbers, and the negative result on sparse TRAINING

**Training speedup: 2.02x, placement-controlled.** Both arms back to back in ONE allocation, same
host, same contiguous GPUs, matched micro-batch, matched 262144 tokens/step: dense **1.538** vs
sparse **0.762** s/step. Cross-validated -- the dense figure matches the live 90k arm's 1.590, the
sparse figure matches 0.776 measured on a different host.

**Do NOT compare s/step across jobs.** The same sparse config measured 0.75-0.80 s/step on one
host and 1.71-1.84 on another, a factor of 2.3, i.e. larger than the effect. Earlier figures of
2.55x / 2.15-2.31x were cross-job and are WITHDRAWN.

**We did NOT lose GPU exclusivity — be precise about this.** `mode=exclusive_process` implies
`j_exclusive=yes`, so no other job can be assigned our devices, and none was. What is shared on a
node regardless is the NVLink/NVSwitch fabric, PCIe, host memory bandwidth and the power envelope.
The slow host had its OTHER four GPUs held by four jobs belonging to OTHER USERS (osieberl x3,
keshavr x1), filling it to 8/8.

Two explanations RULED OUT, both of which I asserted before checking:
* **Not CPU/dataloader starvation** (what §11.7's placement note diagnoses): the slow host was at
  12% CPU utilisation, the fast one at 67%. Occupied CPU is fine.
* **Not allocation topology.** Every arm here gets a non-contiguous GPU set (2,3,4,6 / 0,1,2,4 /
  5,2,3,4) and most run at full speed, so scattering is the norm, not the cause. A `glink=yes`
  suggestion based on this was withdrawn.
What remains is neighbour LOAD rather than neighbour count: s90k later ran at 0.756 s/step on a
host with four neighbours. §11.8's watchdog note independently measured 2.35 -> 6.05 s/step "when
neighbours arrive", which is the same effect and larger.

**The micro-batch trap.** Dense + `fused_experts` OOMs at mbs 4 (the (4, 4096, 71680) intermediate
is 2.19 GiB), so a naive same-allocation probe forces dense to mbs 1, which costs 1.65x by itself
and inflates the ratio to 3.37x. Match the micro-batch or the number is wrong.

**Sparse TRAINING did not reach dense quality.** Matched schedule, tokens/step, device count and
init; loss gap widened monotonically: +0.004 / +0.030 / +0.042 / +0.068 / +0.081 nats at steps
500-900. Two mechanisms alongside:
* **Expert diversity is not controlled.** Output-space repulsion needs all K expert outputs, which
  the sparse path never computes. Of the substitutes, ranked by the OUTPUT alignment they achieve
  at matched steps: full output-space **0.20-0.27** (unavailable), weight-block cosines
  **0.43-0.44** from scratch but **flat at 0.72-0.75** when applied to an already-collapsed arm,
  and a subsampled output-space estimator **0.53 -> 0.74 and rising** (worst). The subsample is
  unbiased in VALUE (within 2%) but that is the wrong property -- its variance is far higher, and
  under Adam a high-variance term is damped relative to its mean. I validated the value and
  shipped a weak regulariser.
* **The proxy collapses on hand-off.** 0.703 immediately after the switch (so the two-phase
  transfer works), 0.46 within 30 steps of routing being handed over, then only +0.033 per 400
  steps. Routing on the proxy moves the energy landscape faster than a candidate-restricted
  objective tracks it.

**What is NOT separable:** "the design fails" vs "it needs a regulariser we did not find".
Reversing an alignment collapse and preventing one are different problems, and the clean experiment
-- a fresh sparse arm with weight-space repulsion from step 0, ~22 min -- was not run.

**Recommendation for the paper:** report inference sparsity (§app:throughput's 18.99x at production
width, untouched), the training implementation as a placement-controlled 2.02x capability that is
EXACT given the selection (4e-16; `sparse_candidates = K` reproduces dense bit-for-bit on a real
checkpoint), and the training-quality failure as a negative result. All three are in
`sec/appendix.tex` §app:accel-train / §app:accel-procedure as of Overleaf `c8d8b63`.

**Bugs found in this work, all mine, three of them silent:** `proxy_route` never read;
`sparse_backproj` never reaching the pydantic config; `_proxy_step` never called in the sparse path
(so `proxy_loss_coef` was a no-op and an arm trained with a FIXED RANDOM router -- alignment 0.698
-> 0.785 and no metric logged); data-dependent index shapes (wedged distributed compile, 0 steps in
8 minutes); `torch.randint` in the compiled forward (hung a job, 14 min/0 steps); duplicate
candidates double-counted after exploration was added. Every one was caught by a run misbehaving,
not by review.

### 12.14 Depth vs width at FIXED FLOPs, 400M scale — width wins at 1k steps, and it cuts against §12.2

**The design.** Three arms, all at **1.17 G MAC/token** in the mixture, d=1024, K=16, top-2, peak lr
2e-3, 131072 tok/step, 4 GPUs each. Iterations share the recurrent block's weights, so holding
FLOPs fixed while adding depth forces width — and PARAMETERS — down:

| arm | iters | I_e | total params | s/step |
|---|---:|---:|---:|---:|
| `d400_it4` | 4 | 17920 | **401M** | **3.18** |
| `d400_it8` | 8 | 8960 | 254M | 3.73 |
| `d400_it12` | 12 | 5952 | 204M | 4.09 |

**Result — `it4` wins at every step, monotonically:**

| step | it4 | it8 | it12 |
|---:|---:|---:|---:|
| 100 | **6.7398** | 6.9977 | 7.2114 |
| 200 | **6.1626** | 6.3610 | 6.4838 |
| 300 | **5.7384** | 5.9468 | 6.0699 |
| 400 | **5.5016** | 5.6543 | 5.7537 |
| 500 | **5.3381** | 5.4704 | 5.5489 |
| ~1000 | **4.7409** @1080 | 4.8069 @1120 | 4.9547 @990 |

At fixed FLOPs, **parameters beat depth**. And `it4` is additionally **22% faster per step** on
identical arithmetic, because iterations are SEQUENTIAL in a launch-bound block (§7.11: 79.5k
kernels/step, GPU-busy 1.71 s against 6.75 s wall). So iso-FLOP is NOT iso-time, and on wall-clock
`it4` wins by more than the loss table shows. **Anyone repeating this must report both axes.**

**But the gap NARROWS**: it4-minus-it8 runs 0.258 / 0.198 / 0.208 / 0.153 / 0.132 at steps
100-500, and it4-minus-it12 goes 0.472 -> 0.211 over the same window. Extrapolated it would cross
somewhere past a few thousand steps, so "width wins" is supported **at ~1000 steps and not beyond**.
The arms were killed at ~1000 steps.

**Tension with §12.2, which is the evidence that put us on T12.** There the 134M T12
(0.66 G MAC/token, 134M params) beat the 400M 4-iteration arm (1.17 G MAC, 401M params) — deeper,
narrower AND smaller winning. Here the ordering reverses. Both can hold: parameters help early,
compute-efficiency pays late, and §12.2's arms ran 15k+ steps against these 1000. **Consequence: do
not read §12.2 as licence to make the 400M cell narrow-and-deep.** It supports T12 at 134M; at 400M
this ablation favours the existing wide-shallow shape (`it4`, and the sandwich's I_e = 15872). The
T12 advantage may be specific to 134M or to long training, and nothing here settles which.

### 12.15 Sandwich: mu recalibrated (43.24 -> 43.36), and a FLOP-share / robustness dissociation

**`iclr_big_hop_sandwich_sink` had the §12.1 bug**: `sinkhorn_iters: 3` with
`sinkhorn_persist_mu: False` and `sinkhorn_mu_iters: 1` while its block runs **4x**
(`layer_iterations [1, 4, 1]`). Trained mu-tilted, evaluated at mu = 0.

Recalibrated with `calibrate_sinkhorn_mu_20260915.py` (64 batches). Mechanically correct: it probed
**4 mu solves per forward** and wrote `sinkhorn_mu` with shape **(4, 16)** — per-iteration, which is
the thing whose absence made the first attempt at this recover nothing.

| | Avg11 | bits/byte | word ppl |
|---|---:|---:|---:|
| before (mu dropped) | 43.24 | 1.0274 | 45.07 |
| **after (mu restored)** | **43.36** | **1.0247** | 44.62 |

**Use 43.36 in any table**, from `unsharded_mucal`. The gain is small: +0.12pp, -0.0027 bpb.

**And that small gain is the interesting part.** The sandwich has **pure-like FLOP concentration**
— its mixture is **97.6%** of per-token FLOPs (2 GPT wrapper layers are 1.6%, energy attention
0.8%) — yet **hybrid-like robustness** to routing corruption: 0.003 nats here against ~0.003 for
hybrids and **1.6-1.8 nats** for pure 8-12 iteration arms (§12.1), and 0.042 bpb for a 4-iteration
PURE arm (§12.2). Two GPT layers worth 1.6% of FLOPs apparently provide enough of a bypass that
corrupting the energy router barely matters.

**Why this could matter more than the 0.12pp.** Proxy-routed sparsity cost the pure T12 **-0.68pp**
Avg11 but the hybrid only **-0.03pp**. If the sandwich sits at the hybrid end of that
routing-sensitivity spectrum — which this mu result suggests — it would be the **best sparsity
target of the three**: pure-like speedup with hybrid-like tolerance of an imperfect router. Measured
sparse speedup on the sandwich is **1.91x at mbs 1** against an analytical 1.88x, and notably it
does NOT need a large micro-batch, because its experts are I_e = 15872 (3.5x the pure T12's 4480)
and sparsity needs wide experts OR many tokens. NOT placement-controlled — separate hosts.

**Untested and cheap (~30 min):** fit the proxy on `unsharded_mucal` and run the proxy-selection
ablation, exactly as done for pure and hybrid. That would say whether the sandwich escapes the
routing sensitivity that made sparse TRAINING fail on the pure arm (§12.13).

### 12.16 PLAN: LR schedule redesign (WSD), and everything in flight as of 2026-09-16 19:20

Written before a context compaction. Self-contained.

#### The proposed shape, and whether it is standard

User's proposal: **fast drop 1e-3 -> 1e-4, plateau a while, then slow decay to 10x smaller (1e-5)
across 90k steps.**

**This is NOT standard WSD, and the difference matters.** Canonical Warmup-Stable-Decay (MiniCPM,
DeepSeek) plateaus at the **PEAK**: warmup -> constant at high LR for 80-90% of the budget -> short
sharp decay over the last 10-20%. The high-LR plateau is the point: it is where the exploration
happens, and the decay merely cashes it in. The proposal instead drops to a LOW plateau early,
which gives that up. What it resembles is **step / multi-stage decay** (BERT-era, some Llama
variants), which is a real practice but a different one.

If the motivation is "the peak looked too high", the simpler equivalent is just **a lower peak**:
warmup straight to 1e-4 then WSD from there is numerically almost the same as a fast 1e-3 -> 1e-4
drop, without needing a new phase. That is testable with the arms already running.

#### What the codebase supports (checked: `lm_engine/optimization/scheduler.py:38-52`)

Exactly **three phases**: `num_warmup_steps` -> `num_constant_steps` -> `num_decay_steps`, with
cosine or linear decay to `lr * lr_decay_factor`. `num_constant_steps` is 0 in every config we have,
so WSD has never been used here, but it needs **no code change**.

The proposal needs FOUR phases (warmup, fast decay, plateau, slow decay) and would need a new
scheduler class. Do not assume it works from config alone.

#### Two implementable options

**A. True WSD, no code change (RECOMMENDED first).** Peak from the sweep below.
```yaml
lr: <peak>                    # 1e-3 or 5e-4, per the sweep
num_warmup_steps: 1000
num_constant_steps: 74000     # ~82% of budget at PEAK -- the stable phase
num_decay_steps: 15000        # ~17%, cosine
lr_decay_factor: 0.1          # 10x drop
# 1000 + 74000 + 15000 = 90000 = num_training_steps  <- PRE-FLIGHT 1
```
Property that matters for a deadline: you can branch off the stable phase at ANY point, run a short
decay, and get a properly annealed evaluable checkpoint. No need to commit to a token budget.

**B. The proposed 4-phase shape**, if A underperforms: add a `MultiStageScheduler` to
`scheduler.py` taking a list of (steps, target_lr) segments. ~30 lines, and it must be tested
against the existing 3-phase behaviour to avoid perturbing live arms.

#### The measurement that decides the peak and the decay, in flight now

* `sw2k_sparse` (1e-3, 90k-shaped decay -> stays at peak) vs **`sw2k_sparse_c10x`** (1e-3, decay
  COMPLETED in 2000 steps). Same everything else. **The step-2000 gap is the ANNEALING BONUS** at
  0.52B tokens = exactly what WSD's decay phase cashes in. Large bonus => WSD is clearly right.
  Caveat: the decay FRACTION here is 90% against WSD's 10-20%, so the transferable quantity is
  WHERE in the decay the gain lands, not its total size.
* `sw2k_sparse_5e4` (5e-4) vs `sw2k_sparse` (1e-3): the PEAK. At step 400 5e-4 was 0.13 BEHIND
  (5.0490 vs 4.9162), which is expected for a lower peak early and says nothing yet about the
  later "stunting". `sw2k_dense_5e4` was KILLED (relaunch-looping, no progress).
* 12.5 already established: LR **floor is inert** (50x range, inside noise), **peak worth ~0.06
  nats**, and **peak 1e-2 diverges ~1 in 3** — confirmed twice more today, including
  `lrt_sparse_5x` which was AHEAD at step 600 (5.8173, grad_norm 0.24) and then went to
  grad_norm 2.1e7 / loss 13.8 by step 1000. Do not use 1e-2.
* **Decay SPEED had never been varied before `c10x`** — every `pure_lr` arm shared a 5-6k schedule.
  It is also the confound behind "the 30k arms drop faster than the 90k ones": at step 9k they sit
  at lr 1.736e-3 against 1.972e-3.

#### Everything running as of 19:20, and what each settles

| job | id | settles |
|---|---|---|
| `swproxy` | 1713960 | **THE GATE.** Sandwich proxy-selection Avg11 vs dense 43.36. Near -0.03pp (hybrid-like) => launch `sw400_sparse`; near -0.68pp (pure-like) => do not, and 12.15's mu-based prediction does not transfer. |
| `sw2k_sparse_c10x` | 1714132 | the annealing bonus (above) |
| `sw2k_sparse`, `sw2k_dense` | 1713574, 1713573 | 1e-3 pair, 90k-shaped |
| `sw2k_sparse_5e4` | 1713742 | peak 5e-4 |
| `s90k_pure_T12_sparse` | 1710081 | 134M sparse arm; at step ~17.5k, loss 3.83, alignment 0.60, proxy 0.766 — all improving |
| `scale32B_boltz_sinkhorn` | 1701525 | 400M hybrid, 32B tokens, untouched |
| `t90k_pure_T12`, `t90k_hybrid_K32top2` | 1704005, 1704004 | dense 90k references |

**`configs/sw400/sw400_sparse.yml` is committed and pre-flighted, NOT launched** — 400M sandwich
sparse, 15000 steps at 262144 tok/step = 3.93B, matching `iclr_big_hop_sandwich_sink` so the delta
against its recalibrated **43.36** is like-for-like. ~6.7 h on 8 GPUs. `renormalize_topk: true` is
MANDATORY there (the fitted proxy's energy R^2 is -48 to -135000; ranking is fine, magnitudes are
not).

#### Instrumentation warning for whoever picks this up

Four monitor-filter failures today, none of them a real problem with a run: a filter that matched
nothing for 30 min on a healthy job; one that fired on a section HEADER instead of a result; one
that read a single failed `bjobs` query as job death and reported a false alarm; and the
`d400_it4_*` glob that also matched `d400_it4_5x_*` and printed two identical rows.
**Confirm anything a monitor reports by job ID before acting on it.**

#### 12.16a CORRECTION — the sandwich proxy-selection result does NOT exist yet

A monitor reported the sandwich at **+0.00pp** (dense 43.36 / bpb 1.0247, proxy-selected 43.36 /
1.0247) and concluded "hybrid-like, launch `sw400_sparse`". **That was an artifact. The gate is NOT
passed. Do not launch on it.**

Both Avg11 lines cite the SAME directory, `unsharded_mucal/` — the DENSE checkpoint — with
timestamps 19:05 and 19:39. `ablate/C_proxysel/` contains **no `harness_results*.json` at all**. So
the second eval re-evaluated the dense model; nothing measured proxy selection.

Identical-to-4-decimals bits/byte is what gave it away: if routing changed for even a few tokens
bpb would move. **Treat an exactly-zero delta as evidence of a plumbing fault, not of robustness.**

Likely mechanism: `eval_harness.py` does NOT honour `--output_path` (found earlier today — it
writes `harness_results_<timestamp>.json` next to the model), and `C_proxysel/model.safetensors` is
a SYMLINK into `unsharded_sparse_r16m512/`, so results did not land where `compute_avg11.py` looks;
it then fell back to the newest results in the tree.

**To redo it properly:** run the harness against `ablate/C_proxysel` and then VERIFY that
`ablate/C_proxysel/harness_results*.json` exists and that `compute_avg11.py` cites that path,
before reading any delta. The proxy tensors themselves are present and correctly shaped —
`proxy_V (4, 16, 1024, 16)`, `proxy_B (4, 16, 512, 16)`, i.e. 4 per-iteration heads x 16 experts —
and `C_proxysel/config.json` correctly has `proxy_route: true, fused_experts: true, proxy_rank: 16,
proxy_iters: 4, sparse_forward: false`. So only the eval bookkeeping is at fault, not the fit.

One genuine byproduct: the harness is **deterministic to 4 decimal places** across two independent
runs of the same checkpoint (43.36 / 1.0247 both times). That is a useful noise floor — it means a
real Avg11 delta of 0.1pp is signal, not run-to-run variation.

#### 12.16b CORRECTION to 12.16a — the PROXYSEL harness OOMed. It was not the symlink.

12.16a guessed the missing result was `--output_path` bookkeeping plus a symlinked
`model.safetensors`. **That guess was wrong.** The real cause, in
`swproxy_1713960.stderr` (visible only after `tr '\r' '\n'` — the progress bars make it one
giant line that defeats `grep`):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.50 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 10.39 GiB is free.
```

`C_proxysel` has `proxy_route: true` with `sparse_forward: false`, so it runs the **dense**
all-K path *plus* the proxy heads. At 400M with `I_e=15872` that does not fit at
`--batch_size 4`, which is what worked for the 134M models. The DENSE arm on the same
checkpoint succeeded at batch 4 in the same job, which is why the loop looked healthy and
`compute_avg11.py` silently fell back to the newest results in the tree.

Resubmitted as **1715589**: `--batch_size 2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
1 GPU, `-W 08:00`, and the job body now `ls`-es `C_proxysel/harness_results*.json` BOTH before
(confirmed absent → clean measurement) and after, so a fallback cannot be mistaken for a result.

The fit itself was fine: per-iteration heads refined to KL 0.016–0.021, top-2 recall 0.836,
top-3 0.926, top-4 0.955, top-8 0.987.

**Generalisable lesson, and it is the second time this bit us:** `|| echo "... FAILED"` in a
shell loop turns a hard crash into a line of stdout, and the NEXT stage then reads whatever
stale artifact is lying around. `compute_avg11.py` globbing for the newest
`harness_results*.json` in the tree is what converted a crash into a plausible number.
**A stage that produces a number must fail loudly, or verify its own output path exists.**

### 12.17 MEASURED: the annealing bonus saturates after a 1.4x LR reduction

This is the answer to "the sharp-drop arm has the lowest loss, how do we bank on that?" —
and it says: **there is nothing to bank. Do not decay early.**

`sw2k_sparse` (1e-3, 90k-shaped decay, so effectively pinned at peak for all 2000 steps) vs
`sw2k_sparse_c10x` (1e-3, decay COMPLETED in 2000 steps). Identical model, data, warmup 200.
100-step means:

| step | at peak | `c10x` | gap | `c10x` lr | reduction from peak |
|---|---|---|---|---|---|
| 400 | 5.0401 | 5.0353 | +0.005 | 9.73e-4 | 1.03x |
| 600 | 4.7333 | 4.7101 | +0.023 | 8.95e-4 | 1.12x |
| 800 | 4.5515 | 4.4359 | **+0.116** | 7.75e-4 | **1.29x** |
| 900 | 4.4517 | 4.3082 | +0.143 | 7.04e-4 | 1.42x |
| 1200 | 4.1921 | 4.0927 | +0.099 | 4.72e-4 | 2.12x |
| 1600 | 4.0492 | 3.9393 | +0.110 | 2.37e-4 | 4.22x |

**The whole ~0.11 nat gain is realised by a 1.4x LR reduction. The further 3x (7.04e-4 ->
2.37e-4) adds NOTHING** — the gap is flat 900->1600, and the 0.143 at step 900 is probably
a noise excursion on a 100-step mean, so quote **0.11**, not 0.14.

So the fast-drop arm does not have a better trajectory. It has the SAME trajectory plus a
one-time, saturating offset. Consistent with 12.5 ("floor is inert", 50x range inside noise)
and with 12.4 (60k steps pinned at the 2e-4 floor bought 0.015-0.019 nats): once the offset
is collected there is no more to earn, and no step size left to earn it with.

Minor confound, does not affect the contrast: `c10x` has `lr_decay_factor: 0.1` vs `0.02` on
the at-peak arm, but the at-peak arm never leaves ~1e-3 inside 2000 steps.

#### The peak: 12.16's "maybe 2e-3 was too high" is NOT supported

`sw2k_sparse_5e4` is behind `sw2k_sparse` (1e-3) at **every** step and the gap is not closing:
step 800 4.6525 vs 4.5515 (0.101), step 1600 4.1231 vs 4.0492 (0.074). Lower peak = uniformly
worse, same direction as 12.5's "peak is worth ~0.06 nats".

And the observation that originally motivated halving 2e-3 -> 1e-3 — "struggled to drop at
first, then dropped fast and steady" — is the **expected signature of a high peak**, not
evidence against it: measured loss = valley-floor progress + a temperature penalty that grows
with lr, so a high peak looks worse early and better late. **Do not lower the peak on
early-loss appearance.** 2e-3 stands unless something at 400M actually diverges.

#### Recommended WSD, and the distinction that matters

Two DIFFERENT runs; do not conflate their schedules.

1. **`sw400_sparse` (15k, 2e-3, cosine, `num_constant_steps: 0`) — DO NOT TOUCH.** It is the
   sparsity ablation against the dense **43.36**. Changing its schedule destroys the
   like-for-like comparison. It is already pre-flighted.
2. **The 90k headline runs** — this is where WSD goes. Drop-in, NO code change
   (`CosineScheduler` already implements WSD when `num_constant_steps > 0`; verified
   `scheduler.py:90-106`):

```yaml
lr: 2e-3
lr_decay_style: cosine
num_warmup_steps: 1000
num_constant_steps: 80000     # 89% at PEAK -- the stable phase
num_decay_steps: 9000         # 10%; 12.17 says the bonus lands within a 1.4x reduction
lr_decay_factor: 0.1          # -> 2e-4; floor is inert per 12.5
# 1000 + 80000 + 9000 = 90000 = num_training_steps   <- PRE-FLIGHT 1
```
`74000/15000` (17%) is the more conservative literature default if 9000 feels tight.

#### Branch a decay off the stable trunk — this is WSD's real payoff on a deadline

`LoadArgs` supports it: the assert in `arguments.py:151-157` only forbids
`load_lr_scheduler: true` with `load_optimizer: false`, so `load_optimizer: true` +
`load_lr_scheduler: false` is legal. Copy a stable-phase checkpoint, run a short decay-only
schedule, and get a properly annealed evaluable model WHILE THE TRUNK KEEPS RUNNING AT PEAK.

```yaml
# decay-only branch: num_training_steps: 4000, num_warmup_steps: 0,
#                    num_constant_steps: 0, num_decay_steps: 4000, lr: 2e-3
load_args:
  load_path: <stable trunk>
  load_optimizer: true          # keep the Adam moments
  load_lr_scheduler: false      # fresh short decay schedule
  load_starting_iteration: false
```
Consequence: **no need to commit to a token budget up front.** Evaluable checkpoints on
demand. NOT yet run end-to-end — check the `load_dataloader_state` interaction before trusting it.

#### Why the 4-phase "drop to 1e-4, plateau, crawl to 1e-5" shape is the worst option

It forfeits valley progress (12.4: a low plateau is nearly inert) AND pre-spends the annealing
bonus (12.17: it is one-time and terminal). It also needs a new scheduler class. Dropped.
On multi-stage/cyclical cosine: restarts (SGDR) and staged decay are real practices, but for a
single fixed budget they are not what people use — staging is for continued pretraining or a
data-mixture change.

#### 12.16c CORRECTION to 12.16b (and 12.16a). The tooling was fine. The MONITOR fabricated the number.

Third pass on the same incident, and this one is checked against the log line by line rather than
inferred. **Both previous diagnoses were wrong about the mechanism.**

* 12.16a blamed `--output_path` plus the `model.safetensors` symlink. **Wrong.**
* 12.16b blamed `compute_avg11.py` globbing the newest `harness_results*.json` in the tree.
  **Also wrong** — `resolve_results_path` (compute_avg11.py:80-92) globs ONLY under the directory
  it is handed and `sys.exit(1)`s if it finds nothing. It never looks outside. It cannot fall back.

What the log actually contains (`swproxy_1713960.stdout`) — exactly TWO Avg11 lines:

```
101: -------- sandwich / DENSE
128: == Avg11 aggregate from .../unsharded_mucal/harness_results_...19-05-38.json ==
129:   Avg11 = 43.36
146: -------- sandwich / PROXYSEL          <- OOMed, no number
247: -------- sandwich / DENSE             <- the script ran a SECOND time
274: == Avg11 aggregate from .../unsharded_mucal/harness_results_...19-39-22.json ==
275:   Avg11 = 43.36
292: -------- sandwich / PROXYSEL
293: sandwich/PROXYSEL HARNESS FAILED
294: sandwich/PROXYSEL AVG11 FAILED
```

`bench_proxysel_one.sh` was invoked TWICE, and both DENSE evals succeeded with the same 43.36
(which is just the 4-decimal determinism of the harness). PROXYSEL failed both times and said so.
**Every component reported correctly.** `compute_avg11.py` printed
`no harness_results_*.json under .../C_proxysel` and exited 1, exactly as designed.

**The fabrication was in the monitor.** Its filter grepped for `Avg11 *=` across the whole stdout,
collected the two DENSE lines, and presented them as "dense 43.36, proxy-selected 43.36, +0.00pp,
launch `sw400_sparse`". The two numbers were identical because they were THE SAME ARM MEASURED
TWICE — which is also why bits/byte matched to 4 decimals, the thing that (correctly) triggered
the distrust.

**The actual lessons, replacing the two wrong ones:**
1. **A monitor filter must anchor each number to its arm label**, never grep a metric name
   globally. `grep "Avg11 ="` cannot tell you WHICH model produced the line.
2. **A script invoked N times produces N of everything.** Any filter that assumes one number per
   arm per job is wrong the moment a submitter loops.
3. This is the **sixth** monitor-filter failure of the session and by far the most costly — it
   came within one step of launching a 3.9B-token 400M run on a delta that did not exist. The
   standing rule in 12.16 ("confirm anything a monitor reports by job ID before acting on it")
   is what caught it. **Keep it.**
4. What 12.16b got RIGHT and still stands: the PROXYSEL harness genuinely **OOMed** (15.50 GiB
   requested, 10.39 free) because `proxy_route: true` with `sparse_forward: false` runs the dense
   all-K path plus the proxy heads, which does not fit at `batch_size 4` for 400M `I_e=15872`.
   That is the real reason there is no proxy number, and it is fixed by `--batch_size 2`.

**Do NOT "fix" `compute_avg11.py`'s globbing or `eval_harness.py`'s `--output_path`.** Neither is
broken. A TODO item to that effect has been removed.

One real defect does remain in `bench_proxysel_one.sh`: `|| echo "... FAILED"` keeps the pipeline's
exit status at 0, so LSF reports "Successfully completed" for a job that produced nothing. Fixed
below — it now tracks failures and exits nonzero.

#### Retry log for the gate, so nobody repeats these

| job | outcome |
|---|---|
| 1713960 | the original. DENSE fine; PROXYSEL **OOMed** at `batch_size 4`. Monitor misread it as +0.00pp. |
| 1715589 | `batch_size 2` + `expandable_segments`. **Preempted (SSUSP) at 295 s** before reaching the requests; killed deliberately to add a cache. |
| 1716267 | added `--use_cache` AND `--cache_requests true`. Died in 23 s: **`--cache_requests` applies its type conversion before the argparse `choices` check**, so the literal `true` arrives as a dict repr and is rejected. Use `--use_cache` alone. GATE-FAIL fired correctly. |
| 1716546 | `--use_cache` only. RUN, GATE-CLEAN confirmed. ~2-4 h for 82639 requests at batch 2. |

### 12.18 c10x endpoint: the bonus holds at ~0.11 out to a 4.2x reduction. WSD design confirmed.

Completion of the 12.17 measurement. `sw2k_sparse_c10x` (1714132) vs `sw2k_sparse` (1713574),
100-step means, gap = at-peak minus decaying:

| step | at-peak | c10x | gap | c10x lr | reduction from peak |
|---:|---:|---:|---:|---:|---:|
| 700 | 4.6278 | 4.5768 | +0.051 | 8.39e-4 | 1.19x |
| 900 | 4.4517 | 4.3082 | **+0.143** | 7.04e-4 | **1.42x** |
| 1100 | 4.2670 | 4.1518 | +0.115 | 5.50e-4 | 1.82x |
| 1200 | 4.1921 | 4.0923 | +0.100 | 4.72e-4 | 2.12x |
| 1400 | 4.1208 | 4.0104 | +0.110 | 3.25e-4 | 3.08x |
| 1600 | 4.0492 | 3.9393 | +0.110 | 2.37e-4 | **4.21x** |

**The gap is FLAT (+0.100..+0.111) from step 1200 to 1600 while the LR falls a further 2x.** Peak
excursion 0.143 at a 1.42x reduction; durable value **~0.11**. Beyond ~1.4x, further decay buys
NOTHING. This is the strongest single justification for the 90k WSD shape: a 10% decay window
reaches a 10x reduction, which is ~7x more than needed to collect the whole bonus.

**Caveat on the numbers past step 1000.** c10x was preempted THREE times and each time resumed
from `global_step1000`, so the log contains overlapping step ranges (resets 1540->1010,
1020->1010, 1210->1010) and the 100-step buckets above MIX segments from different restarts --
same checkpoint, but different data order after each resume. Re-reading it after the third restart
moved the values by <=0.004 and changed no conclusion, but this is NOT one clean trajectory past
step 1000. If it ever needs to be a figure, plot the FIRST segment only (it reaches step 1540).

**Two operational notes.**
1. `submit_selfresuming.sh` WORKS -- three `runtime_resume_*.yml` in the save_path and every
   restart picked up step 1000 rather than 0. This is the mechanism 12.10 was written about, and
   it is now confirmed under real preemption.
2. But c10x is **LIVELOCKED**: `save_interval` 1000 against a 2000-step run means it must survive
   1000->2000 uninterrupted to checkpoint again, and it keeps dying at 1200-1540. It will likely
   never write `global_step2000`. **Its measurement is complete, so it is pure waste of 4 GPUs.**
   General lesson: `save_interval` must be << the remaining run length on a preemptable queue, or
   an arm can burn GPUs indefinitely while making zero net progress. For a 2000-step probe,
   save_interval should have been ~250.

### 12.19 THE GATE RESOLVED: sandwich proxy selection costs -0.40pp. 12.15's prediction is REFUTED.

Job **1716546**, verified properly this time: `ablate/C_proxysel/harness_results_2026-09-16T21-46-40`
exists (33497 bytes) and `compute_avg11.py` cites that exact path. The evaluated `config.json` has
`proxy_route: true, proxy_rank: 16, proxy_iters: 4, proxy_kind: subspace, sparse_forward: false`.

**All three arms measured IDENTICALLY** (`renormalize_topk: False` in both dense and proxy legs, so
the only difference is who selects the experts):

| model | dense Avg11 | proxy-routed Avg11 | delta |
|---|---:|---:|---:|
| `iclr_hop_K32_top2_sink` (hybrid) | 44.58 | 44.55 | **-0.03pp** |
| `iclr_big_hop_sandwich_sink` | **43.36** | **42.96** | **-0.40pp** |
| `pure_hop_T12_sink` | 40.96 | 40.28 | **-0.68pp** |

Sandwich WikiText 47.49 word-PPL vs the dense 44.62, i.e. ~1.0415 vs 1.0247 bpb (+0.017).

**VERDICT: the gate ("near -0.03pp => launch") is NOT passed.** -0.40pp is 13x the hybrid's cost
and 59% of the pure arm's, i.e. squarely in the middle rather than at the hybrid end.

**12.15's dissociation does not generalise.** It observed that the sandwich has pure-like FLOP
concentration (mixture = 97.6% of per-token FLOPs) but hybrid-like robustness to dropping the
Sinkhorn dual (0.003 nats vs 1.6-1.8 for pure stacks), and predicted the sandwich would therefore
tolerate an imperfect PROXY router like a hybrid. It does not. **Robustness to a mis-tilted router
(a shift in mu, shared across experts) and robustness to mis-SELECTION (picking the wrong experts)
are different properties, and the first does not imply the second.** Two GPT layers can carry a
bypass around a mis-scaled energy landscape while still being unable to substitute for the right
expert. Do not reuse the mu-sensitivity number as a proxy-sensitivity predictor for any arm.

#### What this does and does NOT settle

**Settled:** a POST-HOC proxy retrofit onto a trained dense sandwich costs -0.40pp. So
"fit a proxy to the existing 43.36 checkpoint and ship it sparse" is not viable at the -0.03pp
standard the hybrid set.

**NOT settled: from-scratch sparse TRAINING**, which is what `wsd90k_*_sparse` (1717220/1717221)
actually does. Three reasons the retrofit number does not transfer:
1. In a retrofit the model is frozen and the proxy must chase it. In training the model
   CO-ADAPTS to the proxy, and `sparse_explore` supplies exact energies for experts the proxy
   would not propose.
2. The retrofit ran `renormalize_topk: False`, so the proxy had to estimate the all-K denominator
   from energies whose R^2 is **-48 to -135000** (ranking fine, magnitudes unusable). The training
   arms set it TRUE, which deletes that term. 12.13: renormalisation is "free when TRAINED
   (44.54 vs 44.58)" but "+0.483 bpb retrofitted".
3. Consequently **neither** retrofit variant predicts a from-scratch arm: `C_proxysel` carries the
   denominator error, and `Cp_proxysel_renorm` (built, never evaluated) would carry the +0.483 bpb
   renormalisation-retrofit penalty. **Do not spend a GPU on `Cp_proxysel_renorm` expecting a
   verdict** -- it answers neither question.

**So the real gate for the launched arms is their own first ~1000 steps**, not this eval:
* `expert_cos_abs_mean` -- 12.13's ranking says weight-space repulsion should hold **0.43-0.44**
  from scratch. If it climbs toward 0.7, diversity control has failed and 12.13 is repeating.
* `proxy_topk_agree` -- chance is k/p = 2/4 = 0.50 with exploration on. The 134M s90k arm reached
  **0.7628 by step 27840**.
* loss against the dense `sw2k_dense` trajectory at matched steps (5.88 s/step at 4 GPUs).

#### Standing correction to the eval protocol, now proven to work

The verification added after 12.16a/c did its job: two failed attempts printed GATE-FAIL and
`no harness_results_*.json` rather than a plausible wrong number, and the successful one printed a
path that could be checked. **Keep the pattern: assert the output file exists, and require the
aggregator to name the path it read, before quoting any delta.**

### 12.20 SCALING TO 700M / 1B: sizing formula, and it IS reachable before Sep 24 -- with sparse

#### Exact parameter formula, calibrated against real checkpoints (not estimated)

Read off `sandwich400`'s safetensors: moe **260.0M**, embed **102.8M**, attn 9.4M, other 28.3M,
total **400.6M**. So, with `tie_word_embeddings: true` and V = 100352:

```
params(d, I_tot) = d*I_tot  +  V*d  +  9.4M*(d/1024)^2  +  28.3M*(d/1024)^2
                   ^mixture    ^embed     ^attn            ^GPT wrappers etc
```
`d*I_tot` is EXACT for the mixture (1024 x 253952 = 260.05M vs 260.0M measured) because a hopfield
expert carries ONE weight matrix, not two. Predicts 400.5M against an actual 400.6M.
Per-token mixture MACs = `d * I_tot * iterations`, and the mixture is **97.6%** of per-token FLOPs
on the sandwich, so wall clock scales with `d*I_tot` to within a couple of percent.

#### Sizing and wall clock, K=16, 90k steps at 262144 tok/step = 23.59B tokens, sparse at ~1.85x

| target | d | I_e | I_tot | FLOPs vs 400M | 8 GPU | 16 GPU | 32 GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400M | 1024 | 15841 | 253456 | 1.00x | **1.7 d** | 0.8 d | 0.4 d |
| 700M | 1024 | 34151 | 546416 | 2.15x | 3.6 d | 1.8 d | 0.9 d |
| **700M** | **1280** | **25031** | **400496** | **1.97x** | 3.3 d | **1.6 d** | 0.8 d |
| 1B | 1280 | 39679 | 634864 | 3.12x | 5.2 d | 2.6 d | 1.3 d |
| **1B** | **1536** | **30966** | **495456** | **2.93x** | 4.9 d | **2.4 d** | 1.2 d |

**Scale `d`, not `I_tot`.** 700M at d=1280 costs 1.97x against 2.15x at d=1024; 1B at d=1536 costs
2.93x against 3.12x at d=1280. A wider model needs LESS `I_tot` for the same parameter count because
the tied embedding (V*d) absorbs more of the budget -- so the wider option is simultaneously cheaper
in wall clock AND avoids the FFN:Attn imbalance that CLAUDE.md records as having sunk the B-series
("do not scale the iso-param design"). Note the embedding is 15% of a 1B model at d=1536; V=100352
is a large vocabulary at these scales.

Chinchilla-optimal (~20 tok/param) is 8B / 14B / 20B, so 23.59B is at or past optimal for all three
-- 90k steps is a defensible budget at every size, not an under-trained one.

#### Schedule: 5.5 usable days (finish ~Sep 22 for a Sep 24 deadline)

Feasible with the big two at 16 GPUs: the 400M pair (8 GPUs each) lands ~Sep 18-19, then 700M
(~1.6 d) and 1B (~2.4 d) in parallel from ~Sep 19 land ~Sep 21 and ~Sep 22. Peak demand 48 GPUs
against `grp_preemptable` at 736/6144 -- quota is not the constraint, PLACEMENT is (a single-GPU
eval sat PEND 30 min tonight).

**DENSE IS NOT REACHABLE AT THESE SIZES.** 1B dense would be ~4.5 d even at 16 GPUs, and 700M dense
~3 d. Sparsity is what puts 700M/1B inside the deadline at all, which is the strategic answer to
"3 days is too long": at 400M sparse turns 3.1-3.3 d into ~1.7 d, and at 1B it turns ~4.5 d into
2.4 d.

**And WSD means the budget need not be chosen now.** Launch with the 90k shape (stable phase at
peak), and branch a 4k decay off the trunk at whatever step the deadline forces. Report the tokens
actually reached. Under cosine an early stop leaves the model mid-decay and badly annealed; under
WSD every checkpoint on the trunk is one short decay away from being publishable. On a hard deadline
this property is worth more than the ~0.11 nat annealing bonus itself.

#### Unverified at these sizes -- check before trusting the table

* Sparse has never been run above 400M. **But the mbs-4 OOM worry is COMPUTED AWAY** -- see 12.21:
  the capacity buffer is `(K, C+1, I_e)` with `C = ceil(cf*T*p/K)`, so
  `sparse@mbsM / dense@mbs1 = cf * p/K * M = 1.25x` INDEPENDENT of I_e. At 1B that is 4.73 GiB
  against the 3.78 GiB dense mbs 1 would use -- and dense mbs 1 is what already runs everywhere.
  The fallback to mbs 1 / ga 8 remains available (the sandwich still measured 1.83-1.91x there).
* The proxy's quality at larger K*I_e is unknown. The ranking task gets harder as experts multiply;
  `proxy_out_dim: 512` and `proxy_rank: 16` were fitted at I_e ~15-18k.
* The ~1.85x sparse factor is measured on the 400M sandwich and is NOT placement-controlled.

### 12.21 Sparse activation footprint is CLOSED FORM: sparse@mbs4 = 1.25x dense@mbs1, at every size

The capacity dispatch allocates `(K, C+1, I_e)` with `C = ceil(cf * T * p / K)`,
`cf = sparse_capacity_factor` (default **1.25**, energy_ff.py:521/1273), `T = mbs*seq`, and during
TRAINING `p = sparse_candidates + sparse_explore = 4`. So the ratio to a dense mbs-1 forward is

```
sparse@mbsM / dense@mbs1  =  cf * (p/K) * M  =  1.25 * 0.25 * 4  =  1.25x
```

**independent of I_e, d, and model size.** Validated against 12.13's measured number: that section
reports the 134M dense mbs-4 intermediate `(4, 4096, 71680)` at 2.19 GiB, and the formula gives
2.19 GiB.

| shape | I_e | dense mbs1 | dense mbs4 | **sparse mbs4** |
|---|---:|---:|---:|---:|
| pure it4 400M | 17920 | 2.19 GiB | 8.75 GiB (OOMs) | **2.73 GiB** |
| sandwich 400M | 15872 | 1.94 GiB | 7.75 GiB | **2.42 GiB** |
| 700M d=1280 | 25031 | 3.06 GiB | 12.22 GiB | **3.82 GiB** |
| 1B d=1536 | 30966 | 3.78 GiB | 15.12 GiB | **4.73 GiB** |

**Consequences.**
1. `micro_batch_size: 4` under sparsity is roughly as safe as `micro_batch_size: 1` dense, which is
   the configuration every dense arm here already runs. So mbs 4 is NOT a memory gamble at any of
   these sizes, and 12.20's "may OOM at 1B" caveat is withdrawn.
2. It also explains why mbs 4 "fits where dense could not": dense mbs 4 is 4x dense mbs 1, sparse
   mbs 4 is 1.25x. The saving is `cf * p/K` = 0.3125x, i.e. **3.2x**, not the naive `k/K` = 8x,
   because the capacity factor and the exploration candidates both cost memory.
3. Lowering `sparse_explore` to 1 would drop p to 3 and the ratio to 0.94x, and raising cf to 1.5
   would push it to 1.5x. Both are levers if a larger model ever does run tight -- but note
   `sparse_explore` is what lets the proxy train at all, so cut cf first.

### 12.22 `sparse_start_step`: the two-phase schedule now runs in ONE job (code, tested)

**Use this instead of the p1dense/phase-2 config pair.** Commits `99d27a3f` (feature) and
`7c4033db` (recompile test).

**Why it was two jobs, and why that was wrong.** The dense->sparse handoff was done as two configs
sharing a `save_path`, because that needed zero code. But every submission costs queue priority on
a busy LSF, the handoff needed a human to watch `proxy_topk_agree` and launch phase 2, and the cost
recurs at every model size (three more pairs at 700M and 1B). On 2026-09-16 the scheduler was
handing out 5-12 minute RUN windows and this became the dominant time sink.

**What made it cheap.** Two pieces of plumbing already existed:
* `forward()` dispatched on `self.sparse_forward` at RUNTIME (`energy_ff.py:1099`), not at
  construction.
* `pretrain.py:400-403` already calls `set_training_step(global_step)` on every model each step
  (added for the cosreg ramp).

So the change is four small edits: the `sparse_start_step` field
(`config/mlp.py`), explicit forwarding (`mlp_blocks/__init__.py` -- pre-flight 7), the gate plus a
`set_training_step` on the MoE (`energy_ff.py`), and propagation to submodules with the module list
cached on first call (`model_wrapper/pretraining.py`).

**Design properties that matter.**
1. Construction-time asserts still key off `sparse_forward`, so a bad sparse config fails at BUILD
   time even though the first N steps run dense.
2. `_sparse_active` is a plain Python bool, so dynamo guards on it: the flip costs exactly ONE
   recompile.
3. It is derived from `global_step`, identical on every rank, so all ranks flip on the same step
   with NO communication. Rank divergence previously wedged a distributed compile, so this is
   load-bearing, not incidental.
4. Defaults are unchanged: `sparse_start_step: 0` is sparse from step 0 (the old behaviour, correct
   when resuming an already-trained proxy), and `sparse_forward: false` is dense forever with
   `set_training_step` a no-op.

**Tested (CPU, `scripts/test_sparse_start_step_20260916.py`, 11 checks):** the knob resolves onto
`.moe` through `get_mlp_block` with a real `EnergyConfig`; the gate is dense at 499 and sparse at
500; dense steps dispatch to `_forward_fused` and sparse steps to `_forward_sparse`; and in float64
the gated dense phase is **bit-identical (0.00e+00)** to a plain dense arm while the gated sparse
phase is bit-identical to a plain sparse arm -- the gate adds no numerical change.

**Tested (CPU, `scripts/test_sparse_start_recompile_20260916.py`):** under `torch.compile` the
module takes `_forward_fused` before the flip and `_forward_sparse` after, outputs differing. This
rules out the dangerous failure -- dynamo baking `_sparse_active=False` into the graph so the flag
flips, nothing errors, and the run silently stays DENSE for 90k steps. That is the shape of all
three silent bugs in 12.13, so it was the one worth testing.

**Still unverified:** whether the recompile HANGS under FSDP + activation checkpointing at
multi-GPU. That failure is LOUD (0 steps, as in the `fused_experts` 2-node wedge and the
data-dependent-shape wedge), so the first real run detects it within a minute. No dedicated GPU
test needed.

**The one thing a single job gives up: `micro_batch_size`.** It is a dataloader parameter, not a
model attribute, so it cannot be flipped mid-run. Dense needs mbs 1 (dense mbs 4 is 8.75 GiB and
OOMs), so a single job runs mbs 1 throughout and takes 4096 tok/call instead of 16384. Acceptable
at 400M+ because both shapes have wide experts (I_e 15872-17920) and 12.15 measured the sandwich at
1.83-1.91x even at mbs 1. **Do NOT copy this to a small-I_e hybrid shape, where 4096 tok/call
loses outright (0.41-0.49x).** If peak throughput matters more than job count, keep the two-config
pair and use mbs 4 in phase 2.

Configs: `configs/wsd90k/wsd90k_{pure_it4,sandwich}_1job.yml`, `sparse_start_step: 500`.

### 12.23 VALIDATED ON HARDWARE: the in-job dense->sparse switch, at 1.996x

`t32B_sandwich_sparse` (job 1718594, 8 GPUs, single node, `sparse_start_step: 300`):

| step | s/step | phase | proxy_topk_agree | expert_cos_abs_mean |
|---:|---:|---|---:|---:|
| 100 | 6.3308 | dense | 0.4702 | 0.5244 |
| 200 | 6.2853 | dense | 0.7275 | 0.7100 |
| 290 | 6.3032 | dense | **0.7858** | 0.6855 |
| 300 | 6.0345 | switching | 0.7718 | 0.6830 |
| **310** | **3.1588** | **sparse** | 0.6054 | 0.6855 |

**6.3032 -> 3.1588 s/step = 1.996x**, and the torch.compile re-trace at the switch did NOT wedge.
That was the one part of `sparse_start_step` the CPU tests could not reach (12.22), so the feature is
now validated end-to-end -- at 8 GPUs / ONE node. **Multi-node is still unverified** (the 2-node
probe 1718609 could not be scheduled: "requirements for reserving resource (ngpus_physical) not
satisfied: 229 hosts" -- two hosts with 8 free GPUs each is far harder to place than one).

The measured 1.996x lands on 12.13's placement-controlled **2.02x** rather than on the 1.85x that
12.20 projected from the sandwich's mbs-1 figure, so the earlier estimate was conservative. At
3.16 s/step the arm finishes 61035 steps in **~54 h = 2.2 days**.

**The dense phase did exactly what it exists for:** proxy agreement 0.125 (chance) -> **0.7858** by
step 290, clearing the 0.75 threshold, and it got there at step ~200 (0.7275) which is why
`sparse_start_step` was cut from 500 to 300.

**The hand-off dip is real but mild.** Agreement fell 0.7858 -> 0.6054 across the switch. 12.13 saw
0.703 -> 0.46 in the two-job setting, so this is roughly half the damage. Note the FLOOR CHANGES at
the switch and this trips people up: dense agreement is against top-2-of-16 (chance **0.125**),
sparse agreement is measured within the p=4 candidate set (chance **0.50**). 0.6054 against 0.50 is
a much weaker margin than 0.7858 against 0.125, so do not read the dip as "still fine".

**Watch `expert_cos_abs_mean`.** It sits at 0.68, against the 0.43-0.44 that 12.13 says weight-space
repulsion reaches from scratch. Too early to call at step 310 (experts have barely differentiated),
but if it is still ~0.68 at step 3000-5000 then the weight-space choice has NOT delivered and 12.13
is repeating. That is the single most likely way these arms disappoint.

### 12.24 scale32B_boltz_sinkhorn benchmarked: Avg11 49.91 at 30.4B tokens

Job 1718621. Unsharded step **58000** (the arm was stopped at 58130 of 61035 = 95.0% to free its 16
GPUs), then mu-recalibrated, then evaluated. Verified: the number is read from
`unsharded_mucal/harness_results_2026-09-17T00-28-39.json` and `compute_avg11.py` cites that path.

| | value |
|---|---|
| **Avg11** | **49.91** |
| WikiText | 25.23 word-PPL |
| tokens | 58000 x 524288 = **30.41B** |
| shape | 400M HYBRID, d=1536, K=32, top-2, `layer_iterations [1,1,1,1,1,1,6]` |

**This is the best Avg11 in the ICLR record**, against 44.58 for the 134M hybrid and 43.36 for the
400M sandwich -- but it is a DIFFERENT TIER (3x the parameters and ~8x the tokens of the 3.9B grid),
so it is not a like-for-like win over either. Quote it with its token count, always.

**Two caveats that must travel with the number.**
1. It stopped at 95% of its schedule, so the cosine decay never reached its 2e-4 floor. By 12.4
   floor steps buy 0.015-0.019 nats per 45k steps, so the finished run would be marginally BETTER:
   49.91 is a slight underestimate, not an overstatement.
2. mu recalibration was REQUIRED (the config has `sinkhorn_iters: 3` but no `sinkhorn_persist_mu`,
   with a 6x block -- the 12.1 bug). The raw `unsharded` eval is queued in the same job and will
   give the delta at this scale; the sandwich's was +0.12pp and this block runs 6x, so it may be
   larger.

#### 12.23a The switch has a TRANSIENT. Do not judge the repulsion setting inside ~200 steps of it.

I read the first 120 steps after the sandwich's switch as "weight-space repulsion is failing" and
recommended reverting both arms to the subsampled output-space setting. **That was wrong, and the
data reversed within another 90 steps.** Recorded because the shape is reproducible and the wrong
call is tempting.

`expert_cos_abs_mean` (lower = more diverse) on the sandwich, switch at step 300:

```
310 0.6855 | 340 0.7644 | 370 0.8037 | 400 0.8147 | 410 0.8159  <- PEAK
420 0.8125 | 440 0.8078 | 460 0.8044 | 480 0.8012 | 490 0.7969 | 500 0.7764 | 510 0.7743
```
Ten consecutive falls after the peak, i.e. sustained, not noise. `proxy_topk_agree` over the same
window rose 0.4617 -> 0.4930.

**Pure reproduces the identical shape ~100 steps behind**, and its proxy never dropped below the
0.50 candidate-set floor at all: 0.6850 (dense) -> 0.5161 at 350 -> 0.5386 at 390 -> 0.5287 at 400,
with cos still climbing (0.7559 at 400) i.e. still pre-peak.

**Mechanism:** handing selection to the proxy perturbs which experts see which tokens, diversity
degrades while the routing re-equilibrates, and the regulariser then reasserts. It is a transient of
the SWITCH, not a verdict on the regulariser.

**Rule: the earliest honest read on `expert_cos_abs_mean` is ~200 steps after the switch, and the
real comparison against 12.13's 0.43-0.44 target belongs at step 3000-5000.** The project record
already warned about exactly this class of error -- "a step-30 reading of the tau sweep gave the
wrong answer and had to be retracted" -- and this is the same mistake with a different metric.

Matched-age comparison against `s90k_pure_T12_sparse` (subsampled output-space, the setting 12.13
ranked WORST) is still the right yardstick, but quote it at a fair age: at step ~500 the sandwich is
at cos 0.7764 against that arm's 0.7412, a gap of 0.035 and closing, not the 0.073 it appeared to be
at step 400. That arm went on to reach cos 0.5628 / proxy 0.7632 by step 27800, so recovery over
thousands of steps is the documented expectation for either setting.

#### 12.24a The mu delta at 400M/30.4B is +0.05pp -- and it scales with GPT BYPASS, not iteration count

Both legs of job 1718621, each with its results path verified:

| | Avg11 | bits/byte | word-PPL | path |
|---|---:|---:|---:|---|
| **`unsharded_mucal`** | **49.91** | 0.8709 | 25.23 | `harness_results_2026-09-17T00-28-39` |
| `unsharded` (mu dropped) | 49.86 | 0.8720 | 25.34 | `harness_results_2026-09-17T00-57-39` |
| delta | **+0.05pp** | +0.0011 | -0.11 | |

**12.15 and I both guessed wrong.** The prediction was that this delta might EXCEED the sandwich's
+0.12pp because the energy block runs 6x here against the sandwich's 4x, so more iterations should
compound the tilt. It is smaller. Iteration count is the wrong variable:

| arm | GPT layers (bypass) | energy-block iters | mu cost |
|---|---:|---:|---|
| `scale32B_boltz_sinkhorn` (hybrid) | 6 | 6 | **+0.05pp** |
| `iclr_big_hop_sandwich_sink` | 2 | 4 | +0.12pp |
| pure 8-12 iteration arms | **0** | 8-12 | **1.6-1.8 nats** |

Monotone in the number of GPT layers, NOT in iterations. Same ordering as 12.15's FLOP-share /
robustness dissociation: a non-energy path around the mixture makes the model tolerant to a
corrupted router, and the pure stacks have none. **So predict mu sensitivity (and, by 12.19,
proxy-selection sensitivity) from the bypass, not from depth.**

Practical consequence: for hybrids the mu recalibration is nearly optional (+0.05pp is real -- the
harness is deterministic to 4 dp -- but small). For pure arms it is worth 1.6-1.8 nats and is
mandatory. The sandwich sits in between, which is consistent with it also sitting between hybrid
(-0.03pp) and pure (-0.68pp) on proxy-selection cost.

### 12.25 NEGATIVE: the 2-node wedge is NOT fixed. `repulsion_tensor_idx` was not the cause.

Job **1719429**, 2 nodes x 4 GPUs, `fused_experts: true` + `sparse_forward: true` +
`repulsion_tensor_idx: true` + `repulsion_space: weight`. Killed at **790 s**. All three of
ACCEL_FINDINGS' documented wedge signatures reproduced:

| signature | 2026-09-15 wedge | 1719429 |
|---|---|---|
| first log line -> first step | never (17+ min, killed) | **never, 790 s, 0 step lines** |
| inductor cache writes | **0** | **0 in the last 3 min** (76 total, all stale) |
| log frozen on | one dynamo warning | **a dynamo `functools.lru_cache` warning** |

**So the hypothesis in 12.22/`d0f2fef1` was WRONG.** `repulsion_tensor_idx` is labelled in
`config/mlp.py` as "the leading suspect for the fused_experts multi-node hang" and it independently
unstuck a 2-node job that had stalled 13 min -- but with it enabled AND with `repulsion_space:
weight` (which avoids the output-space einsum on data-dependent indices altogether), the wedge
persists. Suspect (2) is eliminated.

**Scope caveat:** the 2026-09-15 wedge was `fused_experts` ALONE; this probe runs `fused_experts` +
`sparse_forward`, so it does not isolate which component hangs. Operationally that does not matter --
our configs require both (`sparse_forward` asserts `fused_experts`) -- but a diagnosis must not
assume the two wedges share a cause.

**Remaining suspects, from ACCEL_FINDINGS:**
1. the single large fused GEMM shape;
3. an FSDP-gather interaction with the `weight_fn` closure that reads `holder.W.weight` inside the
   compiled region.
Bisect in flight: job **1719489**, identical but `repulsion_coef: 0.0`, which removes the repulsion
path entirely. RUNS => repulsion is implicated even with tensor indices. WEDGES => it is (1) or (3).

#### The planning consequence, which is the expensive part

**One node / 8 GPUs is the only validated shape.** And 2x8 could not even be SCHEDULED (job 1718609
sat PEND 1.5 h: "requirements for reserving resource (ngpus_physical) not satisfied: 236 hosts"),
whereas 2x4 placed instantly -- so 16-GPU runs are blocked twice over, by the wedge and by
placement. Cost the ladder at 8 GPUs:

| arm | 8 GPUs, sparse |
|---|---|
| 400M (running now) | ~2.2 d |
| 700M (d=1280, I_e 25031) | ~3.3 d |
| 1B (d=1536, I_e 30966) | ~4.9 d |

With ~7 days to Sep 24, **1B does not fit** alongside the two 400M arms. Options, in order of
preference: (a) diagnose the wedge -- suspects (1) and (3) are one config change apart and each
probe is ~10 min on 8 GPUs; (b) make 700M the top of the ladder; (c) launch 1B anyway and use WSD's
branch-decay (12.20) to harvest whatever step it reaches by the deadline, reporting the tokens
actually trained. (c) is the only option that yields a 1B number at all, and WSD is what makes it
publishable rather than half-annealed.

### 12.26 DIAGNOSED: the 2-node wedge is REPULSION, not the fused GEMM

Bisect job **1719489**: identical to the wedging probe except `repulsion_coef: 0.0`. It **RUNS** at
2 nodes x 4 GPUs -- step 10 at 4.279 s/step, step 20 at 2.970 s/step -- and crossed the
`sparse_start_step` recompile too.

**So `fused_experts` + `sparse_forward` are FINE multi-node.** ACCEL_FINDINGS' suspect (1), "the
single large fused GEMM shape", is **exonerated**: that GEMM is present and running in the bisect.
The hang is in the repulsion path, and `repulsion_tensor_idx: true` does not prevent it (12.25).

This matters beyond the wedge: it means the 1.61x fusion and the ~2x sparsity are NOT
single-node-only capabilities. Only repulsion is.

#### The suspected mechanism, and it merges suspects (2) and (3)

`build_boltzmann_moe` (energy_ff.py:2406-2411) builds the fused spec with

```python
# `weight_fn` is a closure so FSDP re-gathers are picked up
# (same reason the expert W_slice closures exist).
"weight_fn": (lambda: holder.W.weight),
```

and the repulsion branch (energy_ff.py:1177) is

```python
if self.repulsion_space == "weight":
    self._add_repulsion_loss_weight()      # goes through weight_fn -> the SHARDED parameter
else:
    self._add_repulsion_loss(expert_grads) # activations only
```

**Weight-space repulsion reads an FSDP-sharded parameter inside the compiled region**, which needs an
all-gather; at 2 nodes that is an INTER-NODE collective issued from inside a dynamo graph. That is
the deadlock shape the project memory already warned about -- *"the inductor `spmd_check` all_gather
hang that data-dependent MoE routing triggers"*. So ACCEL_FINDINGS' suspects (2) and (3) are probably
ONE mechanism, reachable specifically when `repulsion_space: weight`.

**Decisive test in flight: job 1719508**, identical but `repulsion_space: output` +
`repulsion_subsample: 64`, which routes to `_add_repulsion_loss(expert_grads)` and never touches
`holder.W.weight`.
* RUNS => the culprit is the sharded-weight read, and multi-node is available with output-space
  repulsion.
* WEDGES => repulsion hangs in either space, so the sharded read is not it and the remaining
  candidate is the repulsion graph itself.

#### If output-space wins, this changes the 700M/1B plan

The running 32B arms use `repulsion_space: weight` (12.13 ranks it better for expert diversity from
scratch, and the sandwich is currently vindicating that -- cos 0.6490 at step 1000 against the
subsampled arm's 0.7350). **Those two arms therefore cannot go multi-node, ever, without changing
the regulariser mid-run.** But 700M/1B are not launched yet, so they could take output-space
repulsion and 16 GPUs, halving their wall clock -- at the cost of the weaker diversity control, and
of not being regulariser-matched to the 400M arms.

That is a real trade to decide deliberately, not by default:
* weight-space + 8 GPUs: better diversity, matched to the 400M arms, 1B ~4.9 d (does not fit Sep 24).
* output-space + 16 GPUs: weaker diversity, unmatched, 1B ~2.4 d (fits) -- IF 2x8 can be scheduled,
  which tonight it could not (1.5 h PEND; 2x4 placed instantly).

### 12.27 THE 2-NODE WEDGE IS FULLY LOCALISED: it is WEIGHT-SPACE repulsion, and nothing else

Three probes at 2 nodes x 4 GPUs, identical but for the repulsion setting. This closes a bug that had
been open and undiagnosed since 2026-09-15.

| probe | repulsion | result |
|---|---|---|
| 1719429 | **weight**, coef 2.0 | **WEDGES** -- 790 s, 0 step lines, 0 inductor cache writes, frozen on a dynamo warning |
| 1719489 | **off**, coef 0.0 | RUNS -- 120 steps, 1.38 s/step, 0 NCCL errors, crossed the sparse switch |
| 1719544 | **output** + subsample 64 | RUNS -- step 10/20/30 at 3.77/3.10/3.03 s/step, 0 NCCL errors |

**What is therefore NOT the problem, contrary to ACCEL_FINDINGS' suspects:**
* the single large fused GEMM shape (suspect 1) -- present and running in both healthy probes;
* `fused_experts` as such -- the doc's "validated single-node only" is now **too pessimistic**;
* `sparse_forward`, the capacity dispatch, and the `sparse_start_step` recompile -- all fine at 2 nodes;
* the Python-`random` pair indices (suspect 2) -- 1719429 had `repulsion_tensor_idx: true`.

**What it is.** `energy_ff.py:1177` branches to `_add_repulsion_loss_weight()` for
`repulsion_space: weight`, and that path reaches the expert weights through the closure built at
`energy_ff.py:2411`, `"weight_fn": (lambda: holder.W.weight)` -- whose own comment says it is a
closure "so FSDP re-gathers are picked up". So weight-space repulsion **reads an FSDP-sharded
parameter from inside the compiled region**, which requires an all-gather; across nodes that is an
inter-node collective issued from within a dynamo graph, and it deadlocks. Output-space repulsion
instead consumes `expert_grads` (activations, already local) and does not touch the closure. That
also explains the asymmetry the project memory recorded -- *"the inductor `spmd_check` all_gather
hang that data-dependent MoE routing triggers"* -- as the same class of fault.

#### One methodological trap, and it nearly produced a wrong conclusion

The FIRST output-space attempt (1719508) failed with `ncclRemoteError` on the **first** collective
(SeqNum=1 ALLREDUCE, `last completed work: -1` on every rank) on hosts `p1-r15-n4` / `p2-r22-n1`,
while the repulsion-off probe had run clean on `p1-r18-n3` / `p2-r16-n1`. Read carelessly that is
"output-space also fails". **It is a DIFFERENT failure**: the wedge is a silent hang with zero NCCL
errors and zero inductor writes; this was NCCL erroring loudly before training began -- the
node-fault signature ACCEL_FINDINGS documented for `p4-r10-n4`. Retrying on other hosts
(`p4-r25-n4` / `p3-r03-n1`) ran clean. **Always separate "hung" from "NCCL-errored": they have
different causes and only the first is ours.** `submit_train.sh` now carries a `SUSPECT_HOSTS` list
(distinct from `BAD_HOSTS`) holding those two, with the promotion bar written in -- one failure
isolates a variable, two earns a blacklist.

#### The fix, and the choice it forces

**Real fix (not yet implemented):** hoist the sharded read out of the graph -- materialise
`holder.W.weight` before entering the compiled region and pass it in as a plain tensor, or compute
the weight-space repulsion under `torch.compiler.disable()`. Either removes the in-graph all-gather.
That is a contained change but it touches the hot path of two live 61035-step arms, so it must be
validated on a 2-node probe BEFORE it goes anywhere near them.

**Until then, the operational rule:** weight-space repulsion => single node, 8 GPUs max.
Output-space repulsion => multi-node available.

The two running 400M arms (1718594, 1718598) use weight-space, so they are 8-GPU-bound for their
whole life. 700M/1B are unlaunched and may pick either:

| option | expert diversity | 1B at 90k-equivalent | matched to the 400M arms? |
|---|---|---|---|
| weight-space, 8 GPUs | better (sandwich cos 0.6490 @1000 vs the subsampled arm's 0.7350) | ~4.9 d -- **misses Sep 24** | yes |
| output-space, 16 GPUs | weaker | ~2.4 d -- fits | **no** |
| fix the closure, then weight-space at 16 GPUs | better | ~2.4 d | yes |

The third row is the only one that is both fast and matched, which is the argument for spending an
hour on the fix rather than accepting the trade.

#### 12.27a OPERATIONAL: watch step time on a multi-day run and REQUEUE on a sustained 2x regression

`t32B_pure_it4_sparse` ran at ~3.2 s/step to step ~1100, then settled at **8.7-9.2 s/step** and
stayed there. Not noise, not the model: its exec host `p4-r05-n1` had **17 job slots in use** while
the sandwich's `p5-r20-n1` had **1**, and the two arms are the same size, same GPU count, same code.

Killed and resubmitted (job 1718598 -> **1720770**). It drew `p2-r20-n4` at **2 slots** and returned
to **~3 s/step** immediately, resuming from its step-1200 checkpoint.

**Why this is the only lever.** The watchdog comments already record that every alternative fails
here: `-x` (whole node) is unobtainable -- "requirement for exclusive execution not satisfied: 663
hosts" -- and left both 32B arms PEND; reserving 16 of a host's 96 slots does not schedule
("ngpus_physical not satisfied"); and `select[ut<0.5]` filters only at DISPATCH, so a host chosen at
1 slot drifts to 9. The root cause is that `-n <nnodes>` requests ONE slot per host for EIGHT GPUs,
so the job is cgroup-limited to about one core and the dataloader starves as soon as neighbours
arrive.

**The rule.** A 3x step-time regression is INVISIBLE in the loss curve -- the curve just advances
more slowly in wall-clock -- and it silently converts a 2.2-day run into 6.4 days, which was the
difference between making and missing the Sep 24 deadline. With `save_interval: 200` a requeue costs
~10 minutes. So: log step time, compare against the arm's own early median, and requeue on a
sustained 2x+ regression. Check the host's slot count (`bhosts -l <host>`, the `njobs`/`run`
columns) to confirm it is contention before blaming the code.

**Gap found while diagnosing this:** `sparse_overflow` is NOT appearing in either arm's logged
metrics, so capacity overflow could not be ruled out as a contributor by measurement -- only by the
17-vs-1 slot difference. Worth wiring into the metrics if this recurs.

#### 12.27b RETRACTION x2: `expert_cos_abs_mean` OSCILLATES. And the proxy, not diversity, is the worry.

I have now misread this metric twice in opposite directions, both times by treating one phase of an
oscillation as a trend. Recording the actual shape so nobody repeats it.

`t32B_sandwich_sparse`, every 100 steps (switch at 300):

```
 400 cos 0.8147 <- local MAX     1300 cos 0.6207 <- local MIN     2000 cos 0.7178 <- rising again
 500 0.7764   700 0.7148   1000 0.6490   1200 0.6369   1400 0.6426   1700 0.6743   1900 0.7109
```

* **First error (12.23a):** at step 400 I read the rise 0.68 -> 0.815 as "weight-space repulsion is
  failing" and recommended reverting both arms. Retracted when it fell.
* **Second error:** at step 1000 (cos 0.6490) I said weight-space was "clearly ahead on diversity"
  against the 134M reference's 0.7350. **Also wrong.** That was a downswing, and the reference arm
  oscillates in the SAME band at comparable age -- 0.6367 / 0.6615 / 0.6934 / 0.6586 across steps
  2700-3000. Neither setting is converging toward 12.13's 0.43-0.44 target.

**Rule: `expert_cos_abs_mean` swings ~0.06-0.10 on a few-hundred-step timescale. Quote a WINDOWED
median over >=500 steps, never a single reading, and never compare two arms at single points.**

#### What the data does support, and it points at the PROXY

Both arms use p=4, so the sparse-path chance floor is k/p = 0.50 for both, making this comparable:

| | proxy_topk_agree |
|---|---|
| `t32B_*` arms (weight-space repulsion) | **0.47-0.54** -- sitting ON the chance floor |
| `s90k_pure_T12_sparse` (output-space + subsample) | **0.66-0.68** -- clearly above |

**Our proxy is no better than random at proposing candidates; the reference arm's is.** Loss is
unaffected so far (3.9329 at step 2000 = 1.05B tokens, dropping normally) -- expected, because
`_forward_sparse` computes EXACT energies for the p=4 candidates and re-ranks, and `sparse_explore: 2`
supplies half the candidates from a deterministic rotation, so a useless proxy costs candidate
QUALITY rather than selection correctness.

**Hypothesis, untested: the two choices are COUPLED.** Weight-space repulsion pushes expert WEIGHTS
apart, and the subspace proxy is built from weight subspaces (`B_k = W_k V_k`). So the regulariser
may be actively degrading the structure the proxy learns to rank -- which would mean
`repulsion_space` and `proxy_kind` cannot be chosen independently, and that 12.13's ranking of
repulsion substitutes (measured WITHOUT a trained proxy in the loop) does not transfer to a
proxy-routed arm. Cheap test: one short arm, weight-space repulsion + `proxy_kind: quad` or a
higher `proxy_rank`, and see whether agreement lifts off the floor.

**Watch `proxy_topk_agree`, not `cos`, from here.** If it is still ~0.50 at step 5000 while the
reference sits at 0.68, the proxy is contributing nothing and the sparse arms are effectively running
`sparse_explore`'s deterministic rotation as their router -- which works, but is not the method the
paper describes.

#### 12.27c MULTI-NODE IS ~50% FLAKY AT STARTUP, independently of the wedge

Four 2-node x 4-GPU attempts tonight, four distinct host pairs:

| job | hosts | outcome |
|---|---|---|
| 1719489 | p1-r18-n3, p2-r16-n1 | clean, 120 steps |
| 1719508 | p1-r15-n4, p2-r22-n1 | **`ncclRemoteError` at SeqNum=1** |
| 1719544 | p4-r25-n4, p3-r03-n1 | clean, 120 steps |
| 1720214 | p2-r15-n4, p3-r10-n3 | **`ncclRemoteError` at SeqNum=1**, 8 errors |

**Two of four died on the FIRST collective**, each on a different host pair, with
`last completed work: -1` on every rank. Initially I treated this as bad hosts and added
`p1-r15-n4` / `p2-r22-n1` to a `SUSPECT_HOSTS` list. **That inference now looks wrong**: if a handful
of hosts among ~500 were at fault, drawing them twice in four attempts would be very unlikely. This
reads as general 2-node initialisation flakiness on this cluster, so the suspect list is not the
right tool and was not extended.

**Do not confuse the two multi-node failures. They are different and only one is ours:**

| | wedge (ours) | startup fault (cluster) |
|---|---|---|
| NCCL errors | **0** | **8**, `ncclRemoteError` |
| inductor cache writes | 0 | n/a, dies first |
| symptom | silent hang, log frozen on a dynamo warning | loud failure at SeqNum=1, `last completed work: -1` |
| when | after compile, never reaches step 1 | before training begins |

**Consequence for planning, and it is the important part.** Even with the wedge fixed, a 16-GPU run
has roughly a coin-flip chance of failing at startup and would need babysitting to restart. Combined
with 12.25's finding that 2x8 could not be SCHEDULED at all (1.5 h PEND against 2x4 placing
instantly), **multi-node is not a viable path for the Sep 24 deadline regardless of the wedge.**
Cost the 700M/1B ladder at 8 GPUs / 1 node: 3.3 d and 4.9 d.

The wedge verdict is still worth having for the record and for post-deadline work -- job **1722297**
is the third attempt at it -- but it is no longer on the critical path.

#### 12.27d MY W-REUSE FIX WAS INSUFFICIENT. The wedge is the RESHAPE, and it is inherent.

Job **1722297**, weight-space repulsion at 2 nodes WITH the `adb90ac9` fix: **626 s, 0 step lines,
0 NCCL errors, 0 inductor cache writes** -- the wedge signature, distinct from the cluster startup
fault (which is loud, `ncclRemoteError` at SeqNum=1). So removing the second `_fused_W()` read was
necessary-looking but **not sufficient**, and 12.27's claim that the second read *was* the mechanism
is downgraded to "was one of the mechanisms".

**The remaining cause, and it is not a redundancy this time** (`energy_ff.py`,
`_add_repulsion_loss_weight`):

```python
Wv = W.reshape(self.n_experts, -1)      # regroups rows BY EXPERT
```

With `fsdp_algorithm: 2` (per-parameter sharding, DTensor) `W` is sharded on dim 0 -- the `K*I_e`
dimension. The expert GEMM is happy with that: each rank multiplies its own row slice. But repulsion
must compare experts pairwise, so it has to regroup dim 0 into `(n_experts, I_e*hidden)`, and that
reshape crosses shard boundaries -- DTensor must redistribute. **Weight-space repulsion inherently
requires the whole weight matrix regrouped by expert, so it is inherently a collective**, and passing
`W` in cannot avoid it. That is why output-space repulsion is fine: it consumes `expert_grads`, which
are activations and already replicated.

**Keep the fix anyway.** It is bit-identical (verified, aux loss 0.03364023566246033 both ways) and it
does delete a redundant all-gather of a tensor the forward already held, which is a small win
single-node. It just does not unlock multi-node.

**Candidate real fixes, none attempted -- this is now POST-DEADLINE work** (12.27c already put
multi-node off the critical path: ~50% startup flakiness plus 2x8 being unschedulable):
1. compute the repulsion outside the FSDP forward entirely -- an optimizer-step hook or a
   post-backward callback, where a gather is legal and cheap because the term is O(1) in tokens;
2. compute it on local shards and all-reduce the cosine -- needs the sampled pair's rows co-located,
   which dim-0 sharding does not guarantee, so probably requires choosing pairs per-shard;
3. use `fsdp_algorithm: 1` for weight-space arms, if its sharding leaves the regroup local;
4. accept the current state: **weight-space repulsion => single node.**

**Operational rule unchanged and now well-supported: weight-space repulsion is 8-GPU / single-node.
Output-space repulsion runs multi-node.** Both 400M arms use weight-space and are single-node for
life, which is fine because 8 GPUs is what the deadline plan assumes anyway.

---

# 13. SESSION 2026-09-17 — sign convention VERIFIED, and a repulsion setting I got wrong

**Note: this file is now UNTRACKED** (gitignored 2026-09-17, commit `f83e1839`). It is the local
working record and no longer ships to the public fork. It was public 2026-09-12..17; history retains
it. Credential-scanned clean (no keys/tokens/endpoints); what it holds is cluster hostnames and
internal quota detail.

## 13.1 THE SIGN CONVENTION IS CONSISTENT — theory, code and paper now agree. Verified end to end.

This closes the question three sessions have re-derived. Write `A_i(x) = mean(gelu(W_i x)^2) >= 0`
for the **overlap**. The **state energy** is its NEGATIVE:

```
S_i    = -A_i                       unbounded BELOW (Hopfield / bound-state; deliberate)
Z      = sum_i exp(-S_i/tau)      = sum_i exp(+A_i/tau)
E^FF   = -tau log Z                 the FREE energy (physics sign)
p_i    = softmax(-S_i/tau)        = softmax(+A_i/tau)     -> favours the BEST match
forward= -grad E^FF = -sum p_i grad S_i = +sum p_i grad A_i
p(x)  propto exp(-E^FF/tau) = Z     recovers eq:moe-density
```

**What each expert kind STORES, and why `e_sign` differs by kind** (this is the whole confusion):

| kind | `energy_per_token` returns | equals | needed `e_sign` for `logits = -S_i/tau` |
|---|---|---|---|
| hopfield | `+(gelu(Wx)**2).mean(-1)` = `+A_i` | `-S_i` | **`"pos"`** |
| w1w2 | `-d^{-1/2}(phi * W2x).sum(-1)` = `-A_i` | `+S_i` | **`"neg"`** |

Both land on `logits = -S_i/tau`. That is CLAUDE.md pre-flight 8, and every arm sets it via
`e_sign_override` because **the kind-based defaults at `energy_ff.py:2387/2396` are inverted for BOTH
kinds** (w1w2 defaults "pos" on a stored `-A`, hopfield "neg" on a stored `+A`; both give
`logits = +S/tau`, the worst match).

**Code verified line by line:** `_logits_raw` does `s = -E_k if e_sign=="neg" else E_k` then `/tau`;
the block energy is `-tau*logsumexp(logits)` = `E^FF`; each expert returns
`pref*(gelu_Wx*gelu_prime)@W ∝ +grad A_i` and the mixture is a plain `einsum(p, expert_grads)` with
no overall minus. All three match the table above. **No code change needed. The 2026-09-15 sign fix
was correct.**

**Paper fixed and pushed** (Overleaf `18c1e3d`, `233e50f`, `219fede`, `29f9f1c`, `099f6be`):
`sec:freeenergy` had defined the free energy in a SCORE convention (`e^{+E/tau}`, `F = +tau log Z`),
which contradicted the corrected `eq:eff`; and `eq:moe-update`'s forward sign was the OPPOSITE of what
the code computes. Both fixed, `S_i = -E_i` stated explicitly, and the attention branch made to match
(`S_B = -q·k/sqrt(d_h)`, `E^AT = -tau_A log Z^AT`). Numerically checked.

**Still open (documented, not fixed):** `sec:theory-together` claimed the forward pass "is a descent
step". It is a descent **direction** composed with a LEARNED map, and every arm uses
`energy_proj_type: dual_unconstrained` — **two** unconstrained maps, not the single Pi the paper
draws, with `scale_ff` frozen at 1. Only `psd_anti` would make descent exact. Now stated honestly in
theory + `app:expert-forms`. See 13.3.

## 13.2 ⚠ I SET A REPULSION MODE THE CODE DOCUMENTS AS NOT WORKING. Decide before the arms get deeper.

`config/mlp.py:435`, which I did not read before choosing:

> "The alternative, `repulsion_space: weight`, was **MEASURED not to work**: a 1500-step sweep at
> coef **0.7 and 2.0** showed output alignment **RISING 0.29 -> 0.46**, the signature 11.8 records
> for **NO repulsion**."

**`coef 2.0` weight-space is exactly what I set on both live 32B arms.** I justified it from 12.13's
ranking (weight-block 0.43-0.44 beats subsampled output 0.53-0.74) — but that compares the ABSOLUTE
level and ignores the TREND. Weight-space lands in a decent range while RISING, i.e. it is not
controlling diversity. **CORRECTION (same session, an hour later): the live arms do NOT show that signature.** I claimed
they were "drifting up" from single readings; on WINDOWED MEDIANS they are flat-to-decaying --
sandwich 0.6372 -> 0.6274, pure 0.8096 -> 0.7957. That was my FIFTH single-point misread of this
metric (see 13.7 rule 3, which I then violated in this very subsection).

**And 12.13's ranking is itself wrong about subsample.** It called the subsampled output estimator
"worst, 0.53 -> 0.74 and rising". The long run refutes that: `s90k_pure_T12_sparse` with
`output` + `subsample: 64` went **0.7412 -> 0.5628 monotonically over 27k steps**. The "rising" was an
early-window artifact. At matched step ~5-6.5k the comparison is s90k 0.6412, sandwich (weight-space)
0.6274, pure (weight-space) 0.7957 -- so weight-space is not clearly worse on the sandwich, and only
pure looks poor. Architectures differ, so none of this is clean.

**Three mechanisms, do not conflate them again:**

| mechanism | operates on | multi-node |
|---|---|---|
| Sinkhorn (mu) | **logits** | SAFE |
| repulsion `output` (**the default**) | expert outputs / grads | SAFE |
| repulsion `weight` | expert WEIGHT matrices | **WEDGES** |

**The documented recipe for a SPARSE arm is `repulsion_space: output` + `repulsion_subsample: 64`**
(what `s90k_pure_T12_sparse` used; ~1.2% of a sparse step), NOT weight-space. Switching the two live
arms would (a) use a regulariser that works, (b) make them multi-node capable, (c) make them
repulsion-COMPARABLE to `scale32B_boltz_sinkhorn`, which uses output-space — right now the hybrid is
output-space and the pure/sandwich are weight-space, so the 32B cross-architecture comparison is
CONFOUNDED. Cost at step ~6500/61035 (11%): ~6 h.

**REVISED RECOMMENDATION: LEAVE THEM RUNNING.** The case for switching rests on the `mlp.py:435`
sweep, NOT on these arms' behaviour, and the sandwich is tracking as well as the output+subsample
reference at matched steps. Judge at step ~15-27k against s90k's trajectory (0.5906 at 10k, 0.5762 at
15k, 0.5628 at 27.8k), which the 2-hourly check reaches in about a day. Switching now would spend 6 h
on evidence that does not currently support it. What DOES remain true regardless: weight-space keeps
these two arms single-node-bound, and leaves them repulsion-unmatched against `scale32B` (output-space)
-- a real confound for the 32B cross-architecture comparison that should be stated in the paper rather
than fixed by a restart.

## 13.3 MULTI-NODE: Sinkhorn is PROVEN. The blocker is weight-space repulsion under `fused_experts`.

| configuration | 2-node status |
|---|---|
| Sinkhorn + output-space repulsion, DENSE (no `fused_experts`) | **PROVEN — `scale32B_boltz_sinkhorn`, 58,260 steps, 2 hosts, 30.4B tokens** |
| `fused_experts` + `sparse_forward` + output-space repulsion | RUNS (120-step probe, 1.44 s/step) |
| `fused_experts` + `sparse_forward`, repulsion OFF | RUNS (120 steps, 1.38 s/step) |
| `fused_experts` + **weight-space** repulsion | **WEDGES** — unresolved |

**So the recipe to hand colleagues is Sinkhorn + output-space repulsion. It needs no changes.**

**The wedge is diagnosed but NOT fixed.** `_add_repulsion_loss_weight` does
`Wv = W.reshape(self.n_experts, -1)`; under `fsdp_algorithm: 2` W is sharded on dim 0, so regrouping
BY EXPERT crosses shard boundaries and DTensor must redistribute — an inter-node collective issued
from inside a dynamo graph. My fix (`adb90ac9`, pass the already-gathered W in instead of re-reading
it via the `weight_fn` closure) is **bit-identical but INSUFFICIENT**: job 1722297 still wedged
(626 s, 0 steps, 0 inductor writes). The reshape is inherent. Candidate real fixes, post-deadline:
repulsion in an optimizer-step hook OUTSIDE the FSDP forward (most promising — the term is O(1) in
tokens); per-shard pairs + all-reduce; or `fsdp_algorithm: 1`.
`fused_experts` is therefore **NOT** single-node-only, contrary to ACCEL_FINDINGS — only weight-space
repulsion is. The large fused GEMM is exonerated (present and running in both healthy probes).

**Separately: multi-node is ~50% flaky at STARTUP.** 2 of 4 launches died with `ncclRemoteError` at
the FIRST collective (SeqNum=1, `last completed work: -1`), on 4 DISTINCT host pairs. **Do not
confuse the two failures:** the wedge is a silent hang with ZERO NCCL errors; this is loud, before
training begins. Whether it is our code was NOT determined — the question was raised and not
answered. Note `scale32B` ran 58k steps multi-node without it, which argues cluster/fabric rather
than code, but that is inference, not a test.

## 13.4 hopfield vs w1w2: iso-param IS iso-FLOP. And every 400M/32B arm is already hopfield.

**Measured**, same backbone, K=16: `iclr_w1w2_K16_top2` **0.4255 s/step** (53.28 Btok/day);
`iclr_hop_K16_top2_sink` **0.4195 s/step** (54.10). Hopfield is marginally FASTER. Halving `I_e`
buys w1w2 nothing: hopfield does 2 GEMMs at `I_e=X`, w1w2 does 4 at `I_e=X/2` — both `2*d*X*K`.

**Every 400M and 32B arm is `expert_kind: hopfield`**: `scale32B_boltz_sinkhorn` (K=32),
`t32B_pure_it4_sparse`, `t32B_sandwich_sparse`, `iclr_big_hop_{pure,sandwich}_sink`. `w1w2` appears
ONLY in the 134M grid, via the LEGACY `BoltzmannMoE_Energy_MLP` class. **The scaling story is already
consistently hopfield; nothing needs changing.**

**But the justification changed.** At matched K=16 w1w2 WINS: 43.83/40.00 wppl vs hopfield
43.50/40.71. Hopfield's case is now purely STRUCTURAL — one matrix per expert means 2x the experts at
fixed parameters, and hopfield K=32 = **44.58**, the best energy arm, which beats w1w2 K=16's 43.83.
`w1w2_K32_top2` (iso-param: `I_tot` 8192 vs hopfield's 16384) was found STALLED at 10000/30000 and
restarted as job 1731265; it tests whether that structural advantage survives at K=32. Its interim
**41.63** is one-third budget and must NOT be quoted.

## 13.5 Paper: numbers reconciled, and FOUR claims flipped

All pre-sign-correction numbers replaced (5 `app:frontier` rows x 4 columns, ~20 prose sites, 16
recomputed deltas), all traced to results JSONs. Corrected energy values: K=32 top-2 **44.58**,
K=16 dense **44.40**, K=32 top-1 **44.19**, K=16 top-2 **43.50**, no-fixes **43.40**, renorm 44.54,
pure T12 40.96. Baselines UNCHANGED (Switch 44.83, `gptmoe_last_isoP` 44.43) — they are not
energy-routed.

**Flipped claims, all restated rather than patched:**
1. energy vs energy-free stack: was "parity" (44.43 vs 44.38) -> energy nominally AHEAD 44.58 vs
   44.43 (+0.15pp, inside single-seed noise — stated as parity that no longer needs rounding in our
   favour).
2. **hopfield vs w1w2 at matched K,k: hopfield now LOSES on both** Avg11 and PPL.
3. the router x sparsity 2x2 no longer changes sign — the learned gate leads in both cells.
4. "energy only reaches comparable quality by routing densely" is FALSE — the best energy arm is the
   sparsest.

**Routing-balance figures were pre-Sinkhorn** (user's hypothesis, confirmed). Appendix said eff-K
12.3-12.8 of 16, max share 0.17-0.18 — matches nothing. Measured from each arm's training log
(median, last 40 steps): K=16 **15.18-16.00**, K=32 **30.45-31.50**, max_share **0.041-0.130**,
i.e. exactly what `experiments.tex:172-176` claimed. Fixed with provenance inline. The apparent
three-way disagreement dissolved: `app:collapse` measures a DIFFERENT quantity (offline per-token on
trained checkpoints, its own text says so), so it neither confirms nor contradicts.

Also fixed: the `-0.59pp` vs `-0.99pp` figures are NOT in conflict — 39.04 pre-mucal (`-0.98pp`) and
39.43 post-mucal (`-0.59pp`) against the published 40.02. **Quote -0.59pp.** CLAUDE.md states both.

## 13.6 Infrastructure added this session

* **`scripts/bsub/submit_train.sh`** — THE training submitter. Encodes node shape (>=8 and divisible
  by 8 -> 8/node; multi-node adds `span[ptile=1]` + `blaunch`), no `-x` (unobtainable here),
  `select[ut<0.5]` at 8/node, BAD_HOSTS, run-time `load_args`, and a `gpus_per_node` override (arg 7)
  because 2x8 could not be SCHEDULED (1.5 h PEND) while 2x4 placed instantly. **Never hand-roll a
  training bsub.**
* **`scripts/host_placement_log.sh`** — accumulates host placements + verdicts across requeues.
  Keeps FABRIC_FAULT (property of the host; 2 occurrences -> BAD_HOSTS) apart from CONTENDED
  (neighbours arrived; requeue, NEVER blacklist). Backfilled with 15 placements. It immediately
  showed `p4-r10-n4` is the ONLY host meeting the 2-fault bar, so I **cleared** the SUSPECT_HOSTS
  list I had wrongly added 4 hosts to on one fault each.
* **`sparse_start_step`** (`99d27a3f`, `7c4033db`) — dense->sparse in ONE job. Validated on hardware
  at **1.996x** (6.3032 -> 3.1588 s/step at step 300, no wedge). 11 CPU checks incl. bit-identity.
* **`SIGN_CONVENTION_UNIFICATION.md`** — proposal to store `S_i` and default `e_sign="neg"` for both
  kinds. **Post-deadline**: needs a config-version guard or every existing checkpoint's
  `e_sign_override: "pos"` becomes wrong. Pushed to the fork on branch `docs/sign-convention`.
* **Repo sanitized** (`ead350e2`): `experiments/paths.sh` concentrates machine paths (REPO_ROOT
  self-located, so a colleague's clone needs no edit); removed a published map of Overleaf token
  FILENAMES + an internal hostname; `chmod 600` on two 644 token files. `configs/**`,
  `watchdog_loop.sh` (live job) and `watchdog_jobs.conf` deliberately untouched. Launcher payloads
  verified byte-identical.

## 13.7 Operational rules earned this session

1. **Requeue on a sustained 2x step-time regression.** Happened twice: pure off `p4-r05-n1` at 17
   slots (9.07 s), sandwich off `p5-r20-n1` after 1 -> 5 slots (7.96 s). Invisible in the loss curve;
   turns 2.2 d into 5-6 d. A requeue costs ~10 steps at `save_interval: 200`.
2. **`save_interval` << remaining steps on preemptable**, or an arm livelocks (`sw2k_sparse_c10x`:
   interval 1000 on a 2000-step run, preempted at 1200-1540 three times, never checkpointed again).
3. **Quote a WINDOWED MEDIAN (>=500 steps) for `expert_cos_abs_mean` and `proxy_topk_agree`.** I
   misread them FOUR times by treating a phase of an oscillation as a trend — once alarming, once
   "vindicated", both retracted. The dense->sparse switch has a real transient (~200 steps).
4. **An exactly-zero delta is a plumbing fault, not robustness.** The phantom "+0.00pp" was a monitor
   grepping `Avg11 =` globally and pairing two DENSE evals from two script invocations. Anchor every
   number to its arm label.
5. **`compute_avg11.py` picks the NEWEST `harness_results_*.json` in the tree.** A 09-16 re-eval
   covering only the 11 tasks + wikitext therefore silently drops MMLU/GSM8K. Check which file it
   cites.
6. **Watchdog registry is healthy** — all 44 active entries DONE, resubmit counts peak at 7 vs caps
   of 40-400. But **19 entries are commented out and only 4 say why**; a bare `#name` is
   indistinguishable from a failure (this is how I mistook the deliberately-deregistered
   `w1w2_K32_top2` for a watchdog bug). Date and justify future deregistrations.

## 13.8 IN FLIGHT as of 2026-09-17 07:30

| job | id | state |
|---|---|---|
| `t32B_sandwich_sparse` | 1730874 | RUN, step ~6.5k/61035, 3.15 s/step, `p5-r04-n1` |
| `t32B_pure_it4_sparse` | 1720770 | RUN, step ~6.5k/61035, 3.26 s/step, `p2-r20-n4` |
| `projab_{dual_unconstrained,unconstrained,psd_anti}` | 1731163/64/65 | RUN, step ~500/2500, 4 GPUs each |
| `w1w2_K32_top2` | 1731265 | RUN, resumed 10000/30000, ~9 h |

Both 32B arms: 61035 steps x 524288 tok = **32.0B tokens**, matched step-for-step to
`scale32B_boltz_sinkhorn`. ~2 days each. WSD (2000 warmup / 53000 at peak 2e-3 / 6035 decay).

**Projection A/B** (134M dense, 2500 steps): `psd_anti` vs `unconstrained` isolates the PSD
constraint; `unconstrained` vs `dual_unconstrained` isolates map count (`psd_anti` is ONE map,
`dual` is TWO — a 2-arm test would confound them). Step time IDENTICAL across all three
(0.4212-0.4223 s), so the constraint is computationally free. **Do not read before step 2000** — the
ordering already flipped between steps 100 and 400. Noise floor ~0.038 nats. A null result is a
STRONGER justification for leaving the projection unconstrained than the current post-hoc
"more general dynamics" wording.

Cron `973bbc08` checks both every 2 h (session-only; dies with the session).

## 13.9 OPEN DECISIONS

1. **Switch the two 32B arms to `repulsion_space: output` + `repulsion_subsample: 64`?** See 13.2.
   **NOT recommended any more** -- the arms' own windowed medians do not show the failure signature,
   and the sandwich matches the output+subsample reference at matched steps. Re-judge at step 15-27k.
   The residual issue is a REPORTING one: these arms are repulsion-unmatched against `scale32B`, so
   the 32B hybrid-vs-pure-vs-sandwich comparison carries that confound and the paper must say so.
2. **Was the `ncclRemoteError` ours?** Raised, NOT investigated. `scale32B` ran 58k steps multi-node
   without it, which argues cluster rather than code — inference, not a test.
3. **Ship the multi-node recipe to colleagues**: Sinkhorn + output-space repulsion, dense. Proven.
   Warn them about the ~50% startup flakiness (resubmit; it is not their config).
4. `psd_anti` verdict at step 2000, then replace the `\CC` in `app:expert-forms`.
5. 700M/1B sizing is in 12.20; cost at **8 GPUs** (3.3 d / 4.9 d) unless multi-node startup improves.

> ## 📊 DATAMIX REFERENCE — `configs/cmix/REFERENCE_s8e4_stdmoe_fh2_boltz_topk2.yml`
>
> **That vendored file is THE reference for the datamix.** It is the colleagues'
> `s8e4_stdmoe_fh2_boltz_topk2.yml` (`Bharat-Runwal/dolomite-engine`, branch
> `bsaha/boltzmoe-iclr27`), copied in verbatim on 2026-09-17. Any new run that needs to be
> comparable to a colleague's number takes its `datasets:` block from there, not from our
> older configs.
>
> **The mix differs from ours, and it is the reason our numbers are not comparable to theirs
> on any task:**
>
> | | reference (theirs) | our historical mix |
> |---|---|---|
> | `web-nemotron-cc-hq-p2_0` | 0.35 | 0.5 |
> | `web-nemotron-cc-hq-p2_1` | 0.35 | 0.5 |
> | **`megamath-web-pro_0`** | **0.15** | — |
> | **`finemath-3plus-rewritten_0`** | **0.15** | — |
> | `split` | `99,0.5,0.5` | `99.5,0.5,0` |
>
> **70% web / 30% math against our 100% web / 0% math.** Expect this to move MMLU and
> GSM8K-CoT most, but it is not confined to those — treat EVERY cross-group comparison as
> confounded until the mix is matched. Both math sets are present on disk
> (`megamath-web-pro_0` 52 GB, `finemath-3plus-rewritten_0` 79 GB ≈ 13B and 21B tokens), so
> 0.15 × 32B = 4.8B each stays under one epoch.
>
> Use a **separate `data_cache_path`** (`/proj/dmfexp/nima/.cache/megatron_cmix`) — the
> Megatron blend index is keyed on the mix, so pointing at the 100%-web cache dir either
> collides or silently forces a rebuild. Do NOT write into the colleagues' `cache-bsaha`.
>
> **Datamix-controlled arms live in `configs/cmix/`:**
> - `cmix_134M_recur.yml`, `cmix_400M_pure_it4_sparse.yml`, `cmix_400M_sandwich_sparse.yml`
>   — byte-identical to their `projab_dual_unconstrained` / `t32B_*` parents EXCEPT the
>   datamix and the run names (verified: 0 diff lines outside those). Our recurrence,
>   hopfield experts, Sinkhorn and sparsity are UNCHANGED, so each is a clean A/B for the
>   mix alone.
> - `cmix_1B_stacked_sparse.yml` — the colleagues' architecture (12 DISTINCT layers, **no
>   recurrence**: 8× softmax-attn + std MoE, then 4× energy-attn + Boltzmann MoE) ported to
>   our code. Their `mlp_type: BoltzmannMoE_Energy_MLP` (legacy w1w2, pre-Sinkhorn,
>   pre-sparsity) → `EnergyFF_BoltzmannMoE` + `expert_kind: hopfield` + `e_sign_override:
>   "pos"`.
>
> **Three traps in the upstream file, all corrected in our port:**
> 1. `proj_mode: unconstrained` — **that spelling does not exist in our tree** (0 refs). It
>    was a silent no-op that happened to land on the same default. Ours says
>    `energy_proj_type`.
> 2. Its header states its whole purpose is `energy_scale_mode: sqrt_inv_d` "instead of a
>    learnable temperature" — but **the body never sets `energy_scale_mode`** and does set
>    `temperature: 1.0`, and the field does not exist in our tree either. So the file is not
>    what its header claims. Worth raising with them: any "fixed sqrt(1/d) beats learnable T"
>    conclusion may rest on a config that never enabled it.
> 3. Hopfield stores **one** matrix per expert where legacy w1w2 stores two, so at their
>    `intermediate_size: 76288` the port is **~793M, not the ~1105M** of the upstream file.
>    Restoring 1.1B is either 2× `I_e` (19072) or 2× experts (K=16); K=16 is what our own
>    results favour, but expert count was left at their 8 for the first port.
>
> **`repulsion_space` differs by scale, deliberately.** The `cmix_400M_*` arms inherit
> `weight` from their parents (single-node only — weight-space reshapes a dim-0-sharded
> DTensor by expert and WEDGES on >1 node, silently, with 0 NCCL errors). The 1B port is
> multi-node by construction, so it uses `output` + `repulsion_subsample: 64`.

> ## 🔒 RUN CONSISTENCY FOR THE PAPER — ISO-TOKEN, AND GPU COUNT IS RECORDED, NOT REMEMBERED
>
> **Every paper arm trains on exactly 32.0B tokens on the `configs/cmix/` datamix.** Decided
> 2026-09-17 after an audit found the whole 134M tier at **7.9B** (a quarter of budget) and
> `cmix_134M_gptswitch` at **23.6B** — i.e. the baseline had 3x the tokens of the arms it was
> being compared against. That single defect inverted two reported conclusions:
> at 90000 steps gptswitch looked competitive (Avg11 43.82) and looked *better* on wiki ppl
> (40.58 vs 44.54); re-evaluated at its **step-30000** checkpoint for a matched 7.9B it scores
> **42.31** with ppl **55.45** — so hybrid actually leads by **+2.46pp** and wins on perplexity
> too. Never compare arms without checking the budget first.
>
> **`tokens/step = GPUS x micro_batch_size x gradient_accumulation_steps x sequence_length`, and
> GPUS IS NOT IN THE CONFIG.** It comes from the launcher, so a config alone cannot tell you what
> a run trained on. Three mechanisms now close that hole — use them instead of reasoning:
>
> 1. **wandb run config** (`lm_engine/pretrain.py`): every run logs `world_size`,
>    `gpus_per_node`, `num_nodes`, `micro_batch_size`, `gradient_accumulation_steps`,
>    `sequence_length`, `tokens_per_step`, `total_tokens`, `launch_cmd`, `lsf_job_id`,
>    `lsf_hosts` into the run config, so they are sortable run columns and each run
>    self-documents its own budget.
> 2. **Launch ledger** (`experiments/boltzmann-moe/logs/launch_ledger.tsv`), written by
>    `scripts/bsub/submit_train.sh` on every submission: utc, jobid, name, config, **gpus /
>    nodes / gpus_per_node**, queue, wall, mem, mbs, ga, seq, steps, **tokens_per_step**,
>    **total_tokens**, git commit, config sha256, and the exact command line. The resolved
>    config is snapshotted to `logs/launch_configs/<name>_<jobid>.yml`.
> 3. **`scripts/report_cmix_evals.py`** prints Avg11 / MMLU / GSM8K(flex AND strict) / wiki-ppl
>    per arm, resolves GPU count from the ledger or EMPIRICALLY from the log
>    (`billion_tokens_per_day * 1e9 * step_time / 86400`), and **refuses to assume** — it prints
>    `NOT ISO-TOKEN` with the budget groups whenever arms disagree.
>
> **Do not hand-assemble a results table.** A dropped cell once reported wiki-ppl 40.58 as a
> GSM8K score, and `flexible or strict` silently reported strict-match because a genuine `0.0`
> flexible score is falsy in Python. Both are fixed in `report_cmix_evals.py`; run it.
>
> **The 32B reruns are `configs/cmix/cmix_134M_*_32B.yml`** — 122070 steps x 262144 tok/step,
> mbs 4 / ga 2, **LAUNCH AT EXACTLY 8 GPUs**. They are RERUNS, not resumes: the 30000-step arms
> completed their cosine decay to the LR floor, so extending the schedule and resuming would
> re-raise the LR mid-run. The 7.9B numbers remain valid as a shorter-budget datapoint.
>
> **Eval harness:** the colleague's checkout is `experiments/eval_scripts/lm-evaluation-harness`
> (`lm_eval 0.4.9.2`); site-packages has **0.4.11**. For our tasks the definitions are
> scoring-identical — every diff is a `dataset_path` rename (`openai/gsm8k` vs `gsm8k`) — and
> `gsm8k-cot.yaml`'s filters are byte-identical (`strict-match` on `The answer is (...)`,
> `flexible-extract` on `(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1`). **Report
> flexible-extract**, but record both.
>
> **CHECKPOINT RETENTION: `max_to_keep: 2` everywhere.** `cmix_134M_gptswitch` was the lone
> exception at **95**, which is why its step-30000 checkpoint survived to rescue the iso-token
> comparison — but it also cost **136 GB of the 237 GB** under `results/cmix`, on a filesystem
> at **94% full**. Keep 2, plus landmark checkpoints deliberately preserved for iso-token
> re-evaluation.

---

# 14. SESSION 2026-09-18/19/20 — the FLOP-matched baseline, the surrogate router, and six new arms

> **READ 14.1 AND 14.2 FIRST.** They change a headline claim and record two self-inflicted failures
> that will otherwise be repeated.

## 14.1 THE BASELINE WAS UNDER-PROVISIONED BY 12.9% OF FLOPs, AND FIXING IT REVERSES THE PARITY CLAIM

The `6G1S` Switch baseline applies its MoE block **once** where the energy block is applied **six
times**, so it performs 12.9% fewer parameter-applications per token. Ablation B restores recurrence.

**Avg11 = 11-task accuracy mean in percentage points, HIGHER better. ppl = wikitext word-level
perplexity, LOWER better. FLOPwt = millions of parameter-applications per token. All three rows:
134M, 32.0B tokens, 262,144 tok/step, iso-parameter at ~134.4M total / ~123.4M active.**

| arm | FLOPwt | Avg11 | ppl |
|---|---|---|---|
| `6G1x6E` hopfield sparse (energy) | 141.7 | 44.82 | 41.06 |
| `6G1S` Switch, **-12.9% FLOPs** | 123.5 | 44.87 | 40.19 |
| **`6G1x6S` Switch, +1.1% FLOPs** | **143.2** | **45.43** | **39.96** |

So at iso-FLOP the learned gate LEADS by **0.61pp Avg11 and 1.10 perplexity**. The earlier "parity"
reading held only against the cheap baseline. This is the cleanest comparison in the project (same
skeleton, same recurrence, same parameter budget; only the MoE block type differs). **Report it as
competitive-but-behind, never as parity.** Iso-param for the Switch substitute is `I_s = I_e/3`
(swiglu stores three matrices per expert, hopfield one) — at fixed K,k that matches total, active AND
FLOPs simultaneously. 134M: I_s=341. 400M: I_s=1957.

Also: **block PLACEMENT costs more than the routing mechanism.** `6G1x6E` 44.82 vs `5G1x6E1G` 43.45
is 1.37pp for moving the energy block one position earlier, against 0.05pp between `6G1x6E` and the
cheap Switch. The 134M "sandwich" (`5G1x6E1G`) was MISLABELLED — it is hybrid with the energy block
at position 6 of 7, not a thin-bread/thick-core sandwich. Every other sandwich in the repo is
`1G1x4E1G`. The TRUE `1G1x6E1G` was finally built 2026-09-20 (`abl_C_134M_1G1x6E1G_isototal`,
I_total=53248): iso-TOTAL to -0.00% but **-20.1% ACTIVE** and -5.0% FLOPwt, because concentrating
capacity in one recurrent core activates less per token. It is a parameter-efficiency arm, not a
like-for-like swap.

## 14.2 TWO SELF-INFLICTED FAILURES — DO NOT REPEAT

**(a) NEVER `bstop` a pending GPU job.** Suspending invalidates `CUDA_VISIBLE_DEVICES`: all 16 parked
milestone evals failed on resume with `CUDA unknown error ... Setting the available devices to be
zero`. Kill and resubmit to shed queue load. Cost ~3 h of the token-scaling curve. CAVEAT: a
never-suspended eval later died the same way, so host-level CUDA-init failure is a SEPARATE cause
that suspension merely guarantees — an eval may need two or three attempts, and the watcher's
idempotence is what saves it.

**(b) An inherited `mbs`/`ga` written for 8 GPUs gives HALF budget at 4 GPUs.** Hit three times in one
session (`abl_B_134M_6G1x6S`, `abl_D_134M_6G_dense_isototal`, nearly a third). 4 GPUs needs `ga`
doubled. The launch gate now checks **budget, schedule, structure AND build** before submitting —
checking only the build is what let a 16.0B arm through.

## 14.3 `spmd_check` GUARD WIDENED: `n_nodes > 1` -> `world > 1`

`abl_B_134M_6G1x6S` at **1 node x 4 GPUs** died with `InductorError` from
`post_grad_passes -> spmd_check -> dist.all_gather_object` plus a NCCL collective timeout. The old
comment's claim that "single node is unaffected (all ranks on one host agree)" is **DISPROVEN**: graph
divergence is a property of the MODEL (data-dependent structure), not the host boundary. Its 400M twin
survived only because the old guard already applied at 2 nodes. `spmd_check` is a DIAGNOSTIC — it
detects divergence, it does not prevent it — so disabling it costs only comm/compute overlap
reordering. `DOLOMITE_SPMD_DIAG=1` still restores it fail-fast.

## 14.4 `sinkhorn_persist_mu` APPEARS UNSAFE ON MULTI-NODE

Its data-dependent `int(self._mu_call.item())` inside the compiled region **hung a 2-node arm**: log
dead 29 min, LSF still RUN, no NCCL error — the fused-repulsion wedge signature. Rescued by going
single-node at unchanged budget (4 GPUs x mbs4 x ga4 = 262,144). **Three live multi-node arms carry
persist_mu** (400M hybrid, 400M sandwich, 1B); they are fine, so the trigger is not universal, but the
failure mode is SILENT. First thing to check if a multi-node arm freezes with the job still RUN.

**And rule 9 is WORSE for sparse arms than recorded.** All six `cmix_134M_*_32B_sparse*` configs omit
`sinkhorn_persist_mu`. For a DENSE arm the penalty is ~0.003 nats at 6 iterations. For a SPARSE arm
the dual is solved on the CHEAP ROUTER's logits, so untilted eval changes **which experts run**, not
merely how they are weighted.

## 14.5 THE SURROGATE ROUTER: built, works, and a LINEAR head is refuted

A KL-distilled head can replace the rank-r subspace proxy as the cheap selector. It emits
PSEUDO-ENERGIES (not logits), so `_logits_raw`/`_route` supply the e_sign flip, routing_norm, balance
bias, temperature, Sinkhorn dual, top-k and renormalisation unchanged. Sparse wiring is ONE override
of `_proxy_energies`, so `_forward_sparse` is inherited verbatim — the head NOMINATES p >= k
candidates, the EXACT energy still RE-RANKS to k. Distillation is a true KL against a **detached**
Boltzmann distribution, candidate-restricted in the sparse regime, PRE-mu on both sides so mu cancels
from the gradient and the head is never re-targeted at the dense->sparse handover.

**NEGATIVE RESULT — a linear head cannot rank this energy.** Nomination recall = fraction of the true
top-2 appearing among the nominated p (hopfield, K=16, RANDOM weights, so a lower bound for every
selector):

| selector | p=2 | p=4 | top-1 @ p=2 |
|---|---|---|---|
| linear head | 0.472 | 0.715 | 0.570 |
| mlp h=64 | 0.827 | 0.972 | 0.947 |
| mlp h=256 | 0.898 | 0.992 | 0.983 |
| rank-8 subspace proxy | 0.928 | 0.997 | 0.990 |
| chance | 0.125 | 0.250 | -- |

At its ceiling — an OLS fit scores 0.387. **STRUCTURAL:** a linear map is ODD (`f(-x) = -f(x)`) while
the hopfield energy `mean(gelu(Wx)^2)` is approximately EVEN, so a linear head's top-p region is the
antipodal cone of its bottom-p; it mis-ranks half the input space by construction. Over-selection does
NOT rescue it: it needs p=8 of K=16 (50% density, no sparsity left) to match the proxy at p=2.
**An UNSTRUCTURED head is fine** — mlp h=256 is just as independent of the energy's form and reaches
parity. The problem is linear CAPACITY, not missing energy-like structure.

**Over-selection beats head capacity:** 0.898 -> 0.992 from p=2 to p=4, i.e. p=4 with an imperfect
head beats p=2 with a PERFECT cheap router. Every live sparse arm ran `sparse_candidates == top_k == 2`
(no over-selection); set to 4 on 19 UNLAUNCHED configs only.

**Selector cost, MACs per token, LOWER better; % = fraction of the sparse mixture it gates:**

| selector | 134M | 400M |
|---|---|---|
| proxy m=512 (hopfield, as configured) | 327,680 (10.4%) | 786,432 (3.3%) |
| proxy m=I_e (**FORCED** for w1w2) | 458,752 (14.6%) | 3,530,240 (14.7%) |
| head mlp h=256 | 200,704 (6.4%) | 270,336 (1.1%) |
| head mlp h=64 | 50,176 (1.6%) | 67,584 (0.3%) |

h=256 is ALREADY cheaper than the hopfield proxy at both scales, so there is no reason to shrink it.
**The surrogate's real case is w1w2**, where the m-row subsample is mathematically invalid and the
proxy is forced to m=I_e: 14.7% -> 1.1%, a 13x selector saving.

**CAVEATS:** the recall sweep is **hopfield-only and on RANDOM weights**. The proxy is expected to
GAIN on trained weights as W becomes near-low-rank, a mechanism the head lacks. First trained-weight
number, from the live w1w2 sparse arm: `proxy_topk_agree` climbed **0.1255 -> 0.7033** over 115
readings and had not plateaued at step 1,100 — BELOW the 0.898 random-weight figure, i.e. random
weights did NOT extrapolate, and in the unfavourable direction.

## 14.6 W1W2 SPARSE: the subsample fails for a SIGN reason, and the dispatch is exact

`sparse_forward` was hopfield-only (`energy_ff.py:938`). Now registered for w1w2 too, with the assert
kept meaningful by ALSO requiring the subclass to override `_forward_fused` — the w1w2 builder sets
`weight_fn` to W1, so a plain `BoltzmannMoEFFEnergy` handed `kind="w1w2"` would compute the HOPFIELD
energy on it: wrong model, silently, no shape error.

**Dispatch is EXACT** (oracle proxy 3.78e-16..5.22e-16; gradients 6.02e-16; 4.0-4.8e-16 on the real
134M shape). **The m-row subsample is what fails.** Hopfield's `mean(gelu(z)^2)` sums NON-NEGATIVE
terms, so subsampling m rows is unbiased with error `~1/sqrt(m)`. W1W2's `sum_j phi(u_j) v_j` sums
**SIGNED** terms that cancel — measured `|sum|/sum|term|` = **0.0254** for w1w2 vs **1.000** for
hopfield (against `1/sqrt(I_e)` = 0.0211), so subsample error is `~sqrt(I_e/m)`, i.e. >> 1. A faithful
proxy needs `m = I_e`. **This is exactly why the surrogate head matters for w1w2.**
A second trap, silent: `_svd_refit_proxy` picks the m most energetic rows of `U*S` **per matrix**, but
the w1w2 energy pairs coordinate j of W1x with coordinate j of W2x — independent top-m selection pairs
mismatched coordinates. The row subset MUST be common.

**First w1w2 sparse run (134M, mlp h=256 head, p=4):** loss continuous across the switch (5.5503 dense
at step 290 -> 5.5177 sparse at 310), `effective_n_experts` 14.6/16, **no `sparse_overflow`**.
Step-time **0.3582 -> 0.3234 s/step = 1.108x** against a ceiling `K/(candidates+explore)` = 16/6 =
2.67x, i.e. 41% of ceiling (the hopfield arm got 1.546x at 39% of its 4x ceiling). Over-selection to
p=4 is what cuts the ceiling from 4x to 2.67x — that is the cost side of the over-selection trade.

## 14.7 PARAMETER COUNTS WERE WRONG, AND `active_parameters` ON WANDB IS STILL WRONG FOR ENERGY ARMS

Every hand-computed total in this project was LOW by `n_dense_blocks * d * I`: **a swiglu MLP's `c_fc`
is `2*intermediate_size`** (`mlp.py:789`), so a dense swiglu block costs `3*d*I`, not `2*d*I`. Also
**`energy_attention` is `2*d*d`, not `4*d*d`** (c_attn covers Q and K, V=K, no output projection), and
`energy_proj_rank` defaults to **32, not None**. Corrected, cross-checked two ways (`audit_config` from
YAML matches a meta-device build **to the byte** on five configs; the 1B's 1002.067M matches its
safetensors header's 1002.1M to 0.003%):

| config | TOTAL | ACTIVE | FLOPwt |
|---|---|---|---|
| `cmix_134M_hybrid_32B_sparse` | 134.253M | 123.243M | 141.720M |
| `cmix_400M_hybrid_sparse` | 399.784M | 219.427M | 299.376M |
| `cmix_400M_sandwich_sparse` | 400.333M | 156.539M | 217.196M |
| `cmix_400M_baseline_switch` | 400.031M | 231.697M | 231.697M |
| `cmix1B_12L_gptDense_32B` | 1002.067M | 279.320M | 279.320M |

**RETRACTED:** the 400M sandwich is NOT iso-FLOP with the Switch baseline. 217.196M vs 231.697M, i.e.
**6.3% LESS** compute (-11.2% ex-embedding). The recurrence argument strengthens; the wording
overstated sandwich's compute.

**STILL BROKEN:** `calculate_num_parameters()` (`model_wrapper/base.py:206`) recurses for
`get_num_active_parameters()`, and **only `moe.py:418` implements it**. Counters exist in
`energy_ff_paramcount.py` and are wired, but verify per arm: the 1B previously logged **3.6x** its true
active count, and **the ONE arm reported correctly was the Switch baseline every energy arm is compared
against** — i.e. the error flattered the baseline.

**`n_dominant_experts` never reached wandb**: computed at `energy_ff.py:2130` but all three forward
paths gate `_log_metrics` behind `if not torch.compiler.is_compiling():` (`:1233`/`:1487`/`:1672`) and
every live config sets `torch_compile: true`. Fixed with a traced accumulator + `load_n_used_experts`.
**The Switch MoE has NO metrics path at all** (no `get_metrics`, absent from `train_utils.py:78`'s
isinstance tuple) — and adding `MoE` to that tuple WITHOUT also adding `get_metrics` raises
`AttributeError` on the first logged step of the live baseline, because `track_metrics` calls
`get_metrics()` unconditionally after its `hasattr(..., "pop_load_metrics")` check.

## 14.8 Infrastructure added

| path | what |
|---|---|
| `energy_ff_surrogate.py` | KL-distilled head; sparse selection via ONE `_proxy_energies` override |
| `energy_ff_w1w2_sparse.py` | w1w2 sparse forward, registered; dispatch exact |
| `energy_ff_paramcount.py` | `audit_config` (total/active/flop_weight from YAML, no model build) |
| `scripts/gen_status_table.py` | the 12-arm status table, text + LaTeX; feeds paper `app:status` |
| `scripts/gen_arm_table.py` | per-anchor results with structural names |
| `scripts/health_check_arms.py` | arm health; RUN is the only healthy state, PEND-after-requeue and frozen-step are reported |
| `scripts/eval_milestones.sh` | evaluates the 8/16/24/32B hard-linked anchors, split jobs |
| `scripts/merge_eval_results.py` | rejoins split evals (compute_avg11 reads ONE file, does not union) |
| `scripts/unshard_once.sh` | flock'd unshard; two eval halves race otherwise |

**Evals are SPLIT**: `ev_*` = 13 likelihood tasks, `evg_*` = gsm8k_cot alone. gsm8k generation was
~55-85 min of a >1.5 h job and kept dying inside it. Each half skips independently. The gsm8k output
is named `gsm8k_raw_*.json`, NOT `harness_results_*.json`, because `compute_avg11.py` globs
RECURSIVELY and would otherwise select a gsm8k-only file as a complete eval missing 11 of 14 tasks.

**A bitwise A/B on CPU MUST pin `torch.set_num_threads(1)`** — float64 GEMM reduction order depends on
thread count and produces phantom ~1e-16 diffs.

## 14.9 Paper state

`sec/related.tex` written (2 paragraphs) and wired into `main.tex` after the intro. Its argument: works
defining the energy as a deep network's output (`gladstone2026energybased`, `wang2025equilibrium`,
`hoover2023energy`, `dehmamy2025nrgpt`) pay `O(T)` network traversals for T refinement steps because
every step needs `grad E` through the whole network, AND neither E nor grad E has an inspectable closed
form. Ours is closed-form per expert, so the router is an ALGEBRAIC consequence of the energy rather
than something obtained by differentiating a network. The honest counterweight is stated: a bilinear
energy is far weaker than a transformer's, and we claim no reasoning behaviour.
`wong2026affinity` added to the bib and DISTINGUISHED — it uses Friston's FEP as motivation for
additive gate modifications, retains the learned gate, has no partition function, 4 experts.

`app:status` added at the TOP of the appendix: the 12-arm status table with the ablations marked by
role, a full column glossary, and the caveat that rows compare only WITHIN a scale.
**Theory + appendix reparameterised to inverse temperature** `beta = 1/tau`: 40 occurrences in theory,
5 in appendix, **12 tau KEPT** (the released knob's numeric values — converting 0.35 to 2.857 would
make the paper unmappable onto `temperature: 1.0`). BOTH LIMITS INVERTED and checked against
`F = -beta^-1 log sum exp(-beta E_s)`: `beta -> infinity` is winner-takes-all, `beta -> 0` the average.

**CLAUDE.md gained a rule: LABEL EVERY TABLE** — state the quantity, units, better-direction and what
is held constant, immediately above the table.

---

# 15. SESSION 2026-09-20 (later) — six dead arms, three silent failures, and the confound in the sparse story

> **READ 15.1 AND 15.2 FIRST.** 15.1 is a confound that affects how the expert-form result may be
> stated at all. 15.2 is three ways this project loses work without any error being raised.

## 15.1 "SPARSE" IS TWO DIFFERENT MECHANISMS, AND THE GRID CANNOT SEPARATE THEM

The paper had been writing **sparse** unqualified. There are two, and they are different
contributions:

| name | mechanism | config signature | works for |
|---|---|---|---|
| **sparse(proxy)** | router weight matrices replaced by a rank-`r` subspace projection with a small output dim; selection costs `O(K d r)` not `O(K d I_e)` | `proxy_kind: subspace`, `proxy_rank: 16` | hopfield ONLY |
| **sparse(surrogate)** | a small MLP head KL-distilled to reproduce the RANKING of the exact energies; nominates `p >= k` candidates, the EXACT energy re-ranks to `k` | `surrogate_kind: mlp`, `surrogate_replaces_proxy: true`, `proxy_rank: 0` | both; the only option for w1w2 |

**THE CONFOUND: every hopfield arm is sparse(proxy) and every w1w2 arm is sparse(surrogate) or
dense.** So "w1w2 vs hopfield" and "surrogate vs proxy" are the SAME contrast in the data, and
neither is attributable. The arm that separates them — **hopfield + sparse(surrogate) at 134M** —
does not exist and is now `priority.md` P1.3, ahead of any 400M w1w2 run, because it is strictly
more informative per GPU-hour. The reverse control (w1w2 + proxy) is CLOSED, not merely missing:
§14.6's signed cancellation forces `m = I_e`, where the proxy costs more than the mixture it
exists to cheapen.

`gen_status_table.py` now DERIVES the mechanism from the config rather than taking a hand-written
label, specifically so that this confound cannot be mistaken for an independent design choice. It
also distinguishes a third case, **"dense, surrogate router"** (`use_surrogate: true`,
`surrogate_coef: 1.0`), where the head supplies the mixture WEIGHTS rather than the selection.

**FIRST w1w2 NUMBER (dense, surrogate router), and it is NOT a clean expert-form test.**
`cmix_134M_hyb_w1w2_surrMLP_32B`: **Avg11 44.16 / wiki-ppl 40.35 / MMLU 25.42 / GSM8K 1.59**
against the hopfield sparse(proxy) hybrid's **44.82 / 41.06 / 25.57 / 1.59** — i.e. **-0.66pp
Avg11 but 0.71 BETTER perplexity**. It differs in three ways at once (expert form, dense-vs-sparse,
router), and being dense it evaluates all `K=16` experts per token (`sparse_forward: false` ->
`_forward_fused`), so its true FLOPwt is **~207M against 141.7M, ~46% MORE compute for a lower
Avg11**. **`audit_config` reports 141.0M for it and that figure is WRONG** — it charges `top_k`,
which is meaningless for a dense-masked arm. Do not quote it.

## 15.2 THREE SILENT FAILURES — none raised an error, all lost or would have lost real work

**(a) A SILENT CPU FALLBACK holds GPUs and trains nothing.** `abl_E` (job 1796776, host
`p2-r03-n1`) hit `CUDA unknown error` at init on all 4 ranks and **the trainer did not abort** — it
built `DeviceMesh((pp=1, ddp=4, fsdp=1, tp=1), 'cpu', ...)` and began training a 134M model on CPU.
LSF said `RUN`, 2848 s CPU, 69.5 GB resident, **zero steps in 20 minutes** — indistinguishable from
a slow `torch.compile`, which is what makes it dangerous. Would have held 4 GPUs for 24 h. Same
host-level fault as §14.2a but with **no suspension involved**, so `bstop` is not a prerequisite.
Detect with two greps (`DeviceMesh` must not say `'cpu'`; `CUDA unknown error|No CUDA runtime` must
be 0); recover with `bkill` + resubmit under `SUSPECT_HOSTS=<host>`.

**(b) A COMPLETED ARM SAT UNEVALUATED, AND A COMPLETED EVAL SAT UNREAD.**
`cmix_134M_hyb_w1w2_sparse_surr_32B` was at 122,070/122,070 with **no eval at all**, while
`cmix_134M_hyb_w1w2_surrMLP_32B` had a **finished merged eval nobody had read** — the 44.16 above
was already on disk. `tab:status` showed `--` for both and read as "still running".

**(c) `surrogate_free_proxy` BREAKS EVAL, and one half lies about it.** That flag (default true)
converts `proxy_V/B/V2/B2/bias/scale` into NON-PERSISTENT buffers — correct, nothing reads them
once `surrogate_replaces_proxy` overrides `_proxy_energies` — so they are absent from the
state_dict. When lm_eval passes a `device_map`, accelerate builds under `init_empty_weights`, those
buffers land on **meta**, and `dispatch_model -> model.to(device)` raises
`NotImplementedError: Cannot copy out of meta tensor`. `ev_` exited 1 after 67 s; **`evg_` reported
`DONE` while writing nothing**. Fixed in `eval_harness.py` with a fallback that triggers ONLY on
that error, asserts no meta buffers remain before moving the model, and logs itself. Verified:
`device_map='auto'` fails, plain load leaves 0 meta buffers.

**The "reports DONE, writes nothing" pattern has two confirmed causes and looks identical in both.**
The gsm8k half of this eval failed twice: the meta-buffer bug above, then
`CUDA unknown error ... CUDA_VISIBLE_DEVICES` at load. In both cases LSF said `DONE` and the results
file was absent, because the inner script does not propagate python's exit status. **Never treat
`DONE` as evidence an eval ran; confirm the output file exists.** (A suspected THIRD failure was a
false alarm from the recycled-id trap in 15.2d — attempt 3 ran fine throughout.) `submit_gpu_test.sh` (the submitter EVERY
eval goes through) also passed NO `-R select`, so evals could freely land on a host with broken
CUDA; it now carries the same `BAD_HOSTS`/`SUSPECT_HOSTS` exclusion as `submit_train.sh`.

## 15.2d ⚠ LSF RECYCLES JOB IDS — `bhist -l <jobid>` CAN RETURN A MONTHS-OLD JOB

This produced a WRONG conclusion before it was caught. `bhist -l` for jobs 1797724, 1797745 and
1797747 all returned records dated **`Sat Jun 27 2026`** (5.8 s CPU, 5.1 GB) — a recycled id, not
today's job. Reading a hostname out of those records wrongly accused **`p5-r09-n1`**, which has no
evidence against it and must NOT be excluded. `bjobs -o exec_host` on the LIVE job said
`p2-r17-n1` for the same id, flatly contradicting bhist.

**Attribute a host only from `bjobs -o exec_host` while the job is live, or from the LSF job-summary
line that bsub appends to the job's own `$HOME/bsub_logs/*.stdout`:**

```
Job was executed on host(s) <p2-r03-n1>, in queue <preemptable>, ... at Sun Sep 20 04:41:26 2026
```

That line carries its own DATE, so it is self-verifying — which a bhist record is not. Grepping the
stdout for a bare `pN-rM-nK` pattern is NOT enough either: the same file also contains
`Sender: LSF System <lsfadmin@...>` and `was submitted from host <p4-r01-n4>` (the submitting
shell), so match the `executed on host(s)` line specifically. Always check the DATE on a `bhist` record
before believing any field in it. Done properly, the real culprit is **`p2-r03-n1` with THREE faults
on 2026-09-20** (the abl_E CPU fallback plus both gsm8k CUDA-init deaths), which is what is now in
`BAD_HOSTS`. The commit message on `bfc890ec` names p5-r09-n1 and is WRONG; this section supersedes
it.

## 15.3 SIX ARMS WERE DEAD AT ONCE, AND NOTHING WAS GOING TO NOTICE

`grp_ebm` was found at **8/32** — 24 GPUs free, the first time it has not been at 32/32 (the reason
CLAUDE.md sends evals to `preemptable`), and the 8 in use were our own two ablation arms.
Simultaneously dead: `cmix_400M_hybrid_sparse` (58.7%), `cmix_400M_sandwich_sparse` (25.9%),
`cmix1B_12L_gptDense_32B` (91.8%), `cmix_134M_pure_32B_sparse` (88.5%), and — missed by the first
sweep — **`abl_B_400M_6G1x6S` (58.7%), the FLOP-matched Switch baseline the §14.1 headline requires
at 400M.**

**Two gaps caused this and both are now closed.** `watchdog_jobs.conf` contains only the older
`iclr_*` grid arms, so **no `cmix_*` or `abl_*` arm has automatic resubmission** — they are all
hand-managed. And `health_check_arms.py` hardcoded 8 `cmix_*` names under `configs/cmix/`, so the
four `abl_*` arms were invisible to the only check that would have noticed; it now covers 15 arms
with a `cfg_path()` resolver. **`STAT=RUN` is not progress and `bjobs` is not a health check** —
`<save_path>/latest_checkpointed_iteration.json` is authoritative.

**AND THE STEP LINES GO TO STDERR, NOT STDOUT.** stdout holds only the launcher echo, `ninja:` and
the NCCL banner. Grepping stdout shows nothing and looks exactly like a silent hang — three healthy
400M/1B arms were read as wedged at NCCL init on that basis before the stderr check corrected it.

## 15.4 `hopfield_grad_scale` WAS UNREACHABLE FROM YAML FOR THE NON-MoE CLASS

`_EnergyFFHopfieldArgs` never declared the field and `get_mlp_block`'s `EnergyFF_Hopfield` branch
never forwarded it, so `HopfieldFFEnergy` used `"mean"` whatever the config said — **pre-flight
rule 7 in the wild**, invisible because NO config had ever used `EnergyFF_Hopfield` (0 hits in
`configs/`). At `I=2472` `"mean"` gives prefactor **0.00162 against the MoE's 0.125** at
`I_e=1024`, a **~77x weaker descent step** — the regime `_hopfield_grad_prefactor`'s own docstring
records as "the branch was inert" (`||ffwd_out||` 0.005 vs `||attn_out||` 18.53). **The base-EGPT
ablation would have compared a live MoE branch against a dead single-FFN branch and "proved" the
MoE wins.** Fixed in both files; default stays `"mean"`; resolution verified on the built model.

## 15.5 ABLATION E — the non-MoE base-EGPT control, iso-ACTIVE and iso-FLOP

`configs/iclr_26/ablations/abl_E_134M_6G1x6E_baseEGPT.yml`. Same `6G1x6E` skeleton as
`cmix_134M_hybrid_32B_sparse` — same energy attention, `psd_anti`, 6x recurrence, datamix (the
`datasets:` diff against the parent is EMPTY), schedule and 32.0B budget. Only the energy block's
FFN changes: K=16 hopfield experts + top-2 + Sinkhorn + sparse proxy -> ONE `EnergyFF_Hopfield` at
I=2472. Asks whether routing over a mixture buys anything over one energy FFN of the same per-token
size.

| arm | TOTAL | ACTIVE | FLOPwt |
|---|---|---|---|
| `cmix_134M_hybrid_32B_sparse` | 134.253M | 123.243M | 141.720M |
| `abl_E_134M_6G1x6E_baseEGPT` | 123.241M | **123.241M** (-0.002%) | **141.708M** (-0.009%) |

TOTAL is -8.2% **by construction and that is the point** — the MoE stores 16 experts and applies 2,
carrying 11.0M params it never spends on a token. `audit_config` and a meta-device build agree to
the byte. **I=2472 not 2048:** 2048 = `k x I_e` matches the active EXPERT width but leaves FLOPwt
1.4% low, because the MoE's 327,712 router params are applied to EVERY token — and §14.1 is the
cautionary tale for tolerating a FLOP deficit. 2472 also divides by 8 for bf16 tensor cores.
Chose **hopfield** for the single FFN, not w1w2, so the arm changes ONE variable; w1w2 would be the
literal "old `Energy_MLP`" but would confound mixture-removal with expert-form change.

Caveat when reading its wandb: `_log_norms` is gated behind `not torch.compiler.is_compiling()` and
this arm sets `torch_compile: true`, so `output_norm` — the direct check that the FF branch is live
— will NOT appear. Judge branch health from the loss curve against the hybrid.

## 15.6 Repo state

`configs/iclr_26/` **was committed** (`fd52703a`, 2026-09-19) but had **never been pushed**;
`origin` is already the `nimadehmamy` fork. Pushed 2026-09-20 through `038f3c74`, along with
`abl_C`/`abl_D`/`abl_E` and `cmix_134M_hyb_w1w2_sparse_surr_32B` — the last of which had **trained
to completion while untracked**. New: **`configs/iclr_26/priority.md`**, a P0..P3 ordering with the
reason each arm sits where it does, the GPU accounting and the reallocation rule.

**CLAUDE.md gained two user-instruction rules:** NEW CONFIGS ARE CONFIRMED WITH THE USER BEFORE
`bsub` (five points: datamix proven by an empty diff, parent named and diffed, layer structure and
recurrence stated, token budget, and `audit_config` totals against the comparison arm); and WATCH
EVERY RUN OR RESUME every minute for 10 minutes then every 30 minutes, including the stderr fact
and the CPU-fallback greps.

## 15.7 ⚠ ABLATION D LANDED: A PLAIN DENSE GPT BEATS EVERY MoE ARM AT 134M

`abl_D_134M_6G_dense_isototal` — **six dense GPT layers, no MoE, no energy block, no recurrence**
(verified from the config: `num_layers: 6`, `layer_iterations: [1,1,1,1,1,1]`, only
`softmax_attention` + plain `MLP` at `intermediate_size: 3112`) — is now the **best 134M arm on both
Avg11 and perplexity**.

**All rows 134M at 32.00B tokens, 262,144 tok/step. Avg11 and MMLU are accuracy in percentage points
(HIGHER better); ppl is wikitext word-level perplexity (LOWER better); GSM8K is flexible-extract
exact-match %; ACT/FLOPwt are millions of active parameters / parameter-applications per token.**

| arm | arch | ACT | FLOPwt | Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|
| **GPT-only dense, iso-total (ABL D)** | `6G` | 134.3 | **134.3** | **45.55** | **39.94** | 24.45 | 1.74 |
| Switch, learned gate, FLOP-matched (ABL B) | `6G1x6S` | 123.5 | 143.2 | 45.43 | 39.96 | 24.43 | 2.43 |
| Switch, learned gate, unmatched | `6G1S` | 123.5 | 123.5 | 44.87 | 40.19 | 25.28 | 1.36 |
| energy hybrid, sparse(proxy) | `6G1x6E` | 123.2 | 141.7 | 44.82 | 41.06 | **25.57** | 1.59 |
| w1w2, dense, surrogate router | `6G1x6E` | 123.1 | 141.0 | 44.16 | 40.35 | 25.42 | 1.59 |
| energy, block moved, sparse(proxy) | `5G1x6E1G` | 123.2 | 141.7 | 43.45 | 40.98 | 25.20 | 1.67 |

**It leads the energy hybrid by +0.73pp Avg11 and 1.12 perplexity while spending 5.2% FEWER
parameter-applications per token** (134.3 vs 141.7). It also leads the FLOP-matched Switch by
+0.12pp. §14.1 said the learned gate leads the energy router at iso-FLOP; §15.7 says **no router at
all leads both**, more cheaply, at this scale and budget.

**BUDGET VERIFIED BEFORE REPORTING — abl_D is one of §14.2b's half-budget incidents.** Job 1793120
did run at 131,072 tok/step (16.0B). The COMPLETED run is 1793179, whose log shows **first step = 10
and no `load_path`**: it trained all 122,070 steps from scratch at 262,144 tok/step = **32.00B**, so
it did NOT inherit the half-budget checkpoint and is iso-token with every row above.

**FOUR CAVEATS, and the first is the one that matters most:**
1. **SINGLE SEED, NO ERROR BARS.** The multi-seed arms (`iclr_*_s1234`/`_s7`) have been PAUSED in
   `watchdog_jobs.conf` since 2026-09-12, so seed spread at this scale is UNQUANTIFIED. A 0.73pp gap
   may not be outside noise — and that cuts BOTH ways: it applies equally to §14.1's 0.61pp
   Switch-over-energy lead. **Neither margin should be stated as established without seeds.**
2. **Iso-TOTAL, not iso-ACTIVE.** abl_D carries +9% active params (134.3 vs 123.2) while spending
   5.2% fewer parameter-applications, because a 6-layer stack applied once has FLOPwt = active
   whereas the hybrid's 6x recurrence makes FLOPwt > active. It is cheaper in compute and richer in
   activated capacity at the same total — those pull in opposite directions and the row is not a
   single-variable swap.
3. **MMLU is abl_D's WORST column** — 24.45, the lowest of any 134M arm, against the hybrid's 25.57,
   the highest. The dense arm is not uniformly better, and MMLU is the column the 70/30 math datamix
   was supposed to move.
4. **134M is small, and the routing case is a SCALING argument.** The 400M pair (hybrid + the
   FLOP-matched `abl_B_400M_6G1x6S`) is what tests it; both are running.

**WHY `abl_E` IS NOW MORE INFORMATIVE THAN WHEN IT WAS BUILT.** abl_D removes the energy block AND
the mixture; abl_E (§15.5) removes ONLY the mixture, keeping energy attention, the `psd_anti`
projection and 6x recurrence at iso-ACTIVE and iso-FLOP. Together they decompose the gap:
abl_E vs hybrid isolates **the mixture**, abl_D vs abl_E isolates **the energy block itself**.
Before abl_D landed, abl_E was a routing control; now it is the middle term of a three-point
decomposition and should be treated as P1 rather than a nice-to-have.

## 15.8 `normal`/`grp_ebm` IS **NOT** SAFE FROM PREEMPTION — the `priority` queue outranks it

**This invalidates a standing assumption in CLAUDE.md**, which treats `preemptable` as the risky
queue and `normal`/`grp_ebm` as the protected one. The real queue table:

| queue | PRIO | PREEMPTION |
|---|---|---|
| `priority` | **43** | (outranks everything below) |
| `normal` | 30 | `PREEMPTIVE[preemptable] PREEMPTABLE[priority]` |
| `preemptable` | 20 | `PREEMPTABLE` |

So `normal` preempts `preemptable`, and **`priority` preempts `normal`.** On 2026-09-20 at
**06:33:44-46** four arms died together with **exit code 255** on four DIFFERENT hosts
(`p1-r18-n4`, `p5-r09-n4`, `p3-r08-n1`, `p6-r02-n3`): the 400M hybrid, `abl_B_400M_6G1x6S`, the 1B
at 60,670/61,035 (365 steps from finishing), and `cmix_134M_pure_32B_sparse`. The `priority` queue
had 10 jobs running at the time. Three of the four were on `normal`/`grp_ebm`.

**A simultaneous multi-host exit-255 wave is external preemption, not a bug in our configs.** Do not
go looking for a code cause. Check `bqueues -w` for `priority` activity and relaunch.

**I first blamed our own submissions and was WRONG** — the theory was that the three H arms
submitted to `preemptable` had evicted the `normal` jobs. That is impossible: `preemptable` is PRIO
20 and cannot preempt PRIO 30. Check the PRIO column before attributing an eviction.

**Operational consequence:** there is no queue on this cluster where a long run is safe. The only
real protections are frequent checkpoints (`save_interval`), run-time `load_args` resolution so a
requeue resumes instead of restarting, and noticing quickly. Which makes 15.9 matter.

## 15.9 A FROZEN STEP COUNTER IS OFTEN A REQUEUE, NOT A HANG. CHECK CPU TIME FIRST.

`abl_E` sat at step 4,080 for 9 minutes with `STAT=RUN` — the documented wedge signature — and was
about to be killed. It had been REQUEUED: the replacement process resumed from `global_step4000`, so
the step counter went **backwards** (4,080 -> 4,010) and then climbed again.

| | REQUEUE (leave it alone) | HANG (kill and resubmit) |
|---|---|---|
| CPU time | small, RISING fast from ~0 | large, **FLAT** |
| MEM | climbing from ~2 GB | flat at full size |
| step number | JUMPS BACK to last checkpoint | unchanged |
| `grep -c 'Setting OMP_NUM_THREADS'` | **> 1** | 1 |

Same test distinguishes a long COMPILE from a hang: `abl_G_400M_6G1x6E1x6E` showed 0 steps for 18
minutes while CPU time climbed 5,297 -> 7,798 s in 7 minutes of wall clock (~6 cores busy across 8
ranks). That is the first-ever build of two recurrent energy blocks with fused sparse experts, and it
was healthy.

**Killing a requeued-but-healthy arm discards its redone work, and if its first checkpoint has not
been written yet it restarts from ZERO** — which is how `abl_E` lost 1,770 steps earlier the same
day, killed while PEND at step 1,770 with `save_interval: 2000`. **Check for
`global_step*` before killing anything.**

## 15.10 TWO MORE TRAPS FROM BUILDING THE F/G/H ARMS

**(a) HEAD COUNT IS TIER-SPECIFIC AND A META BUILD DOES NOT CATCH IT.** `abl_F` died in 55 s with
`ValueError: rope_dim should be less than head_dim`, because it was built by reusing the 400M
attention template (16 heads) at d=768: `768/16 = 48 < rope_dim 64`. The 134M tier needs **12
heads** (`768/12 = 64`); d=1024 needs 16 (`1024/16 = 64`). Attention params are `4*d*d` and do NOT
depend on head count, so sizing is unaffected — but note **`AutoConfig.for_model` does not validate
`rope_dim` against `head_dim`**, so the meta-device build that "verified" all six configs passed this
one. A meta build catches structure and parameter errors, NOT per-tier attention geometry.

**(b) MULTI-NODE STARTUP IS ~50% FLAKY AND THE FIX IS ONE NODE.** The 1B relaunch died in 38 s with
`RuntimeError: gloo ... Connection closed by peer` — the startup flakiness of §13.9 item 3, not a
config fault. Resubmitted as **1 node x 8 GPUs** instead of 2 x 4 (drop the `gpus_per_node` argument
to `submit_train.sh` and it picks 8/node when the count divides by 8) and it came up immediately.
For a short job, single-node placement being harder to obtain is a much better trade than a coin
flip on rendezvous.

## 15.11 ⭐ THE 134M DECOMPOSITION: THE ENERGY BLOCK HELPS, THE MIXTURE INSIDE IT DOES NOT

Four arms, measured as the **windowed median `train-lm_loss` over steps 29,500-31,500** (the 8B
crossing is step 30,517 at 262,144 tok/step). n=201 logged points each, so the median's standard
error is ~0.001 against a per-step IQR of ~0.021. LOWER IS BETTER.

| arm | 8B loss | ACTIVE | FLOPwt | contents |
|---|---|---|---|---|
| `abl_D` GPT-only, iso-**TOTAL** | **2.8749** | 134.3M | 134.3M | no energy, no MoE, **+9% active** |
| `abl_E` base-EGPT, **no MoE** | **2.8936** | 123.2M | 141.7M | energy attn + 6x recurrence, ONE energy FFN |
| `cmix_134M_hybrid_32B_sparse` | 2.8948 | 123.2M | 141.7M | energy attn + 6x recurrence + **K=16 MoE** |
| `abl_F` GPT-only, iso-**ACTIVE** | 2.9050 | 123.3M | 123.3M | no energy, no MoE |

**(1) THE BOLTZMANN MoE BUYS NOTHING AT 134M, AND THIS PAIR HAS NO CONFOUND.** `abl_E` vs the
hybrid differ by **0.0012 nats, ~1 sigma**, and they are matched on BOTH active parameters (123.2M)
and FLOPwt (141.7M) — same skeleton, same energy attention, same `psd_anti`, same 6x recurrence,
same datamix, same schedule, same 32.0B budget. The ONLY difference is K=16 routed experts versus a
single energy FFN of the same per-token width. **Replacing the mixture with one monolithic expert
changes nothing measurable.**

**(2) THE ENERGY BLOCK ITSELF DOES BUY SOMETHING.** `abl_E` over `abl_F` is **0.0114 nats (~11
sigma)** — but `abl_E` spends **14.9% MORE** parameter-applications per token (141.7 vs 123.3M),
because its block is applied 6x. The gain is real but not free, and this pair is iso-ACTIVE only.

**(3) `abl_D` IS NOT COMPARABLE and its earlier headline is RETRACTED.** §15.7 reported abl_D
beating every MoE arm (Avg11 45.55 vs the hybrid's 44.82) and flagged it as cutting against the
paper. At 8B abl_D is indeed best (2.8749) — but it is iso-TOTAL, carrying **+9% ACTIVE parameters**
while spending 5.2% FEWER parameter-applications. `abl_F`, the iso-ACTIVE control, is the WORST of
the four (2.9050). **So abl_D's win was an active-parameter artifact, not evidence against the
energy architecture.** abl_D and abl_F bracket the hybrid, and which side you land on depends
entirely on which budget is held fixed. Never quote abl_D without abl_F beside it.

**WHY THIS IS THE EXPECTED RESULT AND WHAT IT IMPLIES.** At 134M the expert bank is 12.58M of
134.25M total (9.4%) and only **1.57M active per token** — ~4% of the 46.17M non-embedding active
parameters, against a 77.07M embedding that is 57% of the model. The MoE has almost no capacity to
express an effect, so (1) is what should have been predicted. At 400M the bank is 192.4M of 297.0M
non-embedding total (65%) and 12.02M of 116.7M active (10.3%). **The 134M tier cannot test the
paper's claim; the 400M tier is where it lives.** This is the measured justification for treating
134M as diagnostic-only (`configs/iclr_26/priority.md`).

CAVEATS: single seed, no error bars (the multi-seed arms remain PAUSED, see priority.md P0.4); 8B
rather than 32B; and loss is not Avg11 — the 32B Avg11 ordering at 134M put abl_D 45.55 > Switch
45.43 > w1w2-sparse 45.01 > Switch-unmatched 44.87 > hybrid 44.82, consistent in direction.

## 15.12 TWO MORE 32B RESULTS, AND A CORRECTION TO THE 1B ROW

| arm | Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|
| `cmix1B_12L_gptDense_32B` **CORRECTED** | **47.79** | **29.46** | 26.50 | 1.74 |
| ~~same row as published in `b9c69b2`~~ | ~~41.78~~ | ~~63.55~~ | ~~24.68~~ | ~~1.90~~ |
| `cmix_134M_pure_32B_sparse` (`1x12E`) | 42.00 | 66.08 | 24.43 | pending |

**The published 1B row was a 2.10B-token result.** See 15.13. Corrected, the 1B is the BEST arm in
the table by Avg11 (47.79 against the 400M Switch's 47.44), not the worst. `tab:status` in Overleaf
still carries the wrong figure and must be re-pushed.

`cmix_134M_pure_32B_sparse` at 42.00 / ppl 66.08 is the weakest 134M arm by a wide margin, consistent
with the long-standing finding that the purely-recurrent design underperforms.

---

# 16. STATE AT 2026-09-21 06:30Z — READ THIS FIRST AFTER A COMPACT

**Deadline 2026-09-24 00:00Z, ~65 h left.** Monitors do NOT survive a session restart; re-arm one.

## 16.1 RUNNING RIGHT NOW (grp_ebm 32/32, all 400M)

| job | arm | step | s/step | ETA to 32B |
|---|---|---|---|---|
| 1811959 | `cmix_400M_hybrid_sparse` **P0 energy** | ~53,500/61,035 | 3.9 | ~8 h |
| 1816581 | `abl_H_400M_6G6E_deep` | ~23,300 | 1.7 | ~18 h |
| 1814745 | `cmix_400M_sandwich_sparse` | ~20,300 | 2.5 | ~28 h |
| 1816582 | `abl_H_400M_6G6S_deep` | ~18,600 | 1.5 | ~18 h |

**When `h400M` hits 61,035: queue its eval with `EVAL_QUEUE=ebm bash scripts/auto_eval_on_finish.sh`.**
That is the last number the 400M headline needs. A PREEMPTED EVAL RESTARTS FROM ZERO (measured), so
evals go to ebm while it has room.

Killed deliberately: `abl_G` pair (handed to bsaha), `abl_C`. Our preemptable fairshare is **0.0030**
(~0.333 fresh) so preemptable barely schedules for us.

## 16.2 RESULTS — 134M COMPLETE (11 arms), 400M PARTIAL

Avg11/MMLU pp higher-better; ppl wikitext word-level lower-better; GSM8K flex-EM %; all 32.0B.

| 134M arm | ACT | FLOPwt | Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| `abl_E` base-EGPT **no MoE** | 123.2 | 141.7 | **45.87** | 41.00 | 24.85 | 2.27 |
| `abl_D` GPT-only iso-TOTAL | 134.3 | 134.3 | 45.55 | **39.94** | 24.45 | 1.74 |
| `abl_B` Switch FLOP-matched | 123.5 | 143.2 | 45.43 | 39.96 | 24.43 | **2.43** |
| `abl_F` GPT-only iso-ACTIVE | 123.3 | 123.3 | 45.01 | 41.87 | **25.77** | 2.20 |
| w1w2 sparse(surrogate) | 123.1 | 141.0 | 45.01 | 40.70 | 24.90 | 1.82 |
| Switch unmatched `6G1S` | 123.5 | 123.5 | 44.87 | 40.19 | 25.28 | 1.36 |
| **hybrid `6G1x6E` (headline)** | 123.2 | 141.7 | 44.82 | 41.06 | 25.57 | 1.59 |
| `abl_I` w1w2 uncon-proj | 123.1 | 140.8 | 44.32 | 41.09 | 23.73 | 2.05 |
| w1w2 dense head-as-router | 123.1 | 141.0 | 44.16 | 40.35 | 25.42 | 1.59 |
| block moved `5G1x6E1G` | 123.2 | 141.7 | 43.45 | 40.98 | 25.20 | 1.67 |
| pure recurrent `1x12E` | 86.1 | 185.1 | 42.00 | 66.08 | 24.43 | 1.74 |

| 400M/1B arm | ACT | Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|---|
| **`abl_B_400M` Switch FLOP-matched** | 219.7 | **48.24** | **28.27** | **26.74** | **3.26** |
| `cmix1B_12L_gptDense_32B` | 279.3 | 47.79 | 29.46 | 26.50 | 1.74 |
| `cmix_400M_baseline_switch` unmatched | 231.7 | 47.44 | 28.82 | 26.55 | 2.73 |

**8B windowed-median `lm_loss` (steps 14.5k-16k), 400M:** abl_B 2.6571 · H_6G6S 2.6583 ·
H_6G6G(12G dense) 2.6812 · H_6G6E 2.7280 · hybrid 2.7319 · **sandwich 2.9461**.

## 16.3 THE FOUR FINDINGS THAT MATTER

1. **At 134M, iso-active AND iso-FLOP, the Boltzmann MoE ≈ a single energy FFN.** `abl_E` vs hybrid:
   loss 2.8936 vs 2.8948 (~1σ), ppl 41.00 vs 41.06. Avg11 differs 1.05pp, which with two continuous
   metrics tied is most likely Avg11 noise. **No confound in that pair.** Expected, because the bank
   is 4% of non-embedding ACTIVE at 134M.
2. **`abl_D`'s apparent win is an ACTIVE-PARAM artifact — RETRACTED as evidence.** It is iso-TOTAL
   (+9% active, −5.2% FLOPwt). Its iso-ACTIVE twin `abl_F` is the WORST of the four. Never quote
   abl_D without abl_F.
3. **`psd_anti` BEATS `unconstrained`.** One-field A/B, both 32B: 45.01/40.70/24.90 vs
   44.32/41.09/23.73. **The 6σ 8B loss lead for unconstrained INVERTED by 32B.** So "decide at 8B" is
   a triage rule only — never promote an arm on an 8B delta.
4. **SCHEDULE MISMATCH I INTRODUCED.** `cmix_400M_*`, `abl_B_400M`, `cmix1B` are WSD
   (2000/53000/6035); all 134M arms and **my `abl_G`/`abl_H`** are pure cosine (2000/0/rest), because
   my generator recomputed the schedule instead of inheriting it. Cross-group 8B numbers are
   confounded by ~10% LR. Within-pair (H vs H, G vs G) is clean.

## 16.4 CONFIG TREE — `configs/iclr_26/`

- `scaling/` 22 paper configs + README with full results, and the list of the 11 arms/12 files still
  on `psd_anti` where the field is live (9 others are energy-free, field inert).
- `launch/launch_paper_arms.sh` — all 21 arms, pinned GPU counts, dry-run default, `ALL_400M`.
- `bsaha/` — abl_G pair (8,200 / 21,400), abl_C (60,000), all resumable, + README.
- `bharat/` — **MAIN: `bharat_1B_18L_allMoE_{boltz,switch_baseline}`**, d=1536 L=18, **ALL 18 FFNs
  MoE**, K=32 k=8 I_e=768 so `k·I_e = 4.0×d` (OLMoE practice), softmax attn everywhere so **only the
  router differs**. 1022.4M/512.8M vs 1004.4M/494.8M. Both meta-build OK.
  `bharat/fallback/` — the 8S4E pair (too dilute: 4 of 18 MoE, `k·I_e`=10-23×d).

## 16.5 PUBLISHED CONFIGS (fetched from config.json, not memory)

Dense 1B-class: TinyLlama-1.1B 22L d2048 I5632 vocab32k · Phi-1.5 24L d2048 I8192 ·
Pythia-1.4B 24L d2048 I8192 · SmolLM2-1.7B 24L d2048 I8192. **All have ZERO MoE layers.**
Real MoE: **OLMoE-1B-7B 16L d2048 ALL-16-MoE K64 k8 I_e1024 → k·I_e=4.0×d** ·
Qwen1.5-MoE-A2.7B 24L d2048 ALL-24-MoE K60 k4 I_e1408 (+shared) → 2.8×d.
**Field practice: MoE in EVERY layer, `k·I_e` ≈ 4×d.** Our arms invert this (few, very wide).
This retires our old worry that `I_e=1024` is too narrow — OLMoE uses exactly that.
Our existing 1B: K=64 k=2, 4 of 12 layers, activates **3.1%** of its 746M bank → 279M active, only
+27% over the 400M arms, and it LOSES to abl_B_400M. That is why it "barely beats 400M".

## 16.6 SPEED / EXPERT-FORM VERDICTS (134M, same budget)

`hopfield+proxy 0.20 s/step` · `w1w2+surrogate 0.30` · `w1w2 dense 0.50` · `GPT-only 0.20`.
**The PROXY is 1.5× FASTER than the surrogate.** w1w2+surrogate is nominally better (45.01 vs 44.82,
ppl 40.70 vs 41.06) but inside noise. **Recommend hopfield+proxy.**

## 16.7 OPERATIONAL RULES EARNED TODAY (all now in CLAUDE.md)

- **`normal`/`grp_ebm` is NOT preemption-proof**: `priority` queue is PRIO 43 vs normal 30 and
  preempted 4 arms simultaneously (exit 255, 4 different hosts, within 2 s). A simultaneous
  multi-host exit-255 wave is external preemption — do not hunt for a config cause.
- **A wedged job ignores `bkill`; use `bkill -r`** and VERIFY it left the queue. A corpse held 8 GPUs
  for 20 min and blocked its own replacement.
- **Wedge detection: stderr MTIME, not a CPU delta.** LSF samples CPU coarsely; a healthy arm read
  0 s/25 s while logging. Healthy 0-2 min stale; the real wedge was 34 min. And an arm that has NEVER
  logged a step is COMPILING (25-35 min normal) — use a per-arm baseline: h400M normally reaches its
  first step in ~100 s, so 47 min meant stuck, and killing it was right.
- **A frozen step counter is often a REQUEUE**: step number jumps BACK, CPU resets, >1
  `Setting OMP_NUM_THREADS` banner. Leave it alone.
- **Sub-60 s multi-node deaths are TRANSIENT (~50%)**: gloo `Connection closed by peer`,
  `lsb_launch(): Failed`, wandb `ServiceStartTimeoutError`. Resubmit; export
  `WANDB__SERVICE_WAIT=300`.
- **`sinkhorn_persist_mu` wedges multi-node** — such arms must be single-node (`abl_G` 4-GPU variant).
- **Step lines are in STDERR.** Anchor log globs on `*_${jobid}.stderr`; an arm-name glob returned a
  17,000-step-stale number from a previous run.
- **Never trust `DONE`** — confirm the results file exists. Two evals reported DONE having written
  nothing.
- **Milestones are now IN THE TRAINER** (`SaveArgs.token_milestones_b`, exact tokens/step, before
  pruning). Confirmed in production: 24.01B and 32.00B anchors on abl_B_400M. The old cron inferred
  tokens/step and hard-linked mislabelled anchors (a `tok8B` dir holding 32B weights); 15 were
  renamed `orphan_*`. Backstop runs inside `watchdog_loop.sh`, which self-resubmits.
- **A stale-step eval reached the paper**: `gen_status_table`/`auto_eval` globbed `unsharded*` and
  matched ANY step, so a 2.10B-token checkpoint was published as the 1B's 32B row (41.78 vs the real
  47.79). Both now require the end-of-run checkpoint.

## 16.8 OPEN / NEXT

1. **`h400M` 32B eval** — the last headline number. Queue with `EVAL_QUEUE=ebm`.
2. **Seeds (priority.md P0.4)** — every margin except abl_E-vs-abl_F is sub-1pp on ONE seed,
   including §14.1's Switch lead. `seed` lives at `random_args.seed` (default 42) and is NOT set in
   any config. A 134M arm is ~5.3 h on 4 GPUs.
3. **Overleaf** `tab:status` pushed at `7e5395e` with 19 arms. Repo pushed through `1a777b68`.
4. Decide whether to rebuild `abl_H` on WSD (~24-27 h each) or state the confound.

### 16.2 Eval-on-finish is now UNATTENDED (2026-09-21 16:00Z)

The eval trigger used to be a Claude-session Monitor. A session exit that morning killed four
watches at once, and **nothing unattended covered these arms**: the long-running
`boltz_auto_eval` babysitter drives `collect_flops_wave_20260912.sh`, whose `CFGDIRS` are the
eight `configs/iclr_*` dirs and include **neither `configs/cmix` nor `configs/iclr_26`**. So
`cmix_400M_hybrid_sparse`, `cmix_400M_sandwich_sparse` and the `abl_H` pair would have finished
and sat unscored.

Two independent callers of `auto_eval_on_finish.sh` now exist. It is edge-triggered internally
(fires only when `latest_checkpointed_iteration == num_training_steps`, the results file is
absent, and no live bsub job of that name exists), so two callers cannot double-submit.

1. **`scripts/auto_eval_on_finish_loop.sh`** — CPU-only, `-J boltz_eval_finish`, preemptable,
   300 s cycle, self-resubmitting at 84000 s. First instance: job **1825457** on `p2-r01-n2`.
   Log: `scripts/auto_eval_on_finish_loop.log`.
2. **`watchdog/watchdog_loop.sh`** — same call added next to the milestone backup. NOTE this
   takes effect only on the watchdog's **next self-resubmit** (it had 13:19 left at 16:00Z),
   which is why (1) exists: h400M finishes BEFORE that.

**Queue is preemptable deliberately — do NOT set `EVAL_QUEUE=ebm` in either caller.** grp_ebm is
32 GPUs and routinely 32/32; an eval submitted there pends indefinitely. `auto_eval_on_finish.sh`
already splits `gsm8k_cot` into its own job so a preemption costs one task.

Verified 16:00Z: all four repo configs read `num_training_steps: 61035` and point at the live
save_paths, and a manual driver invocation submitted nothing (correct — no arm finished yet).

**Measured ETAs at 15:55Z** (achieved wall-clock s/step over the preceding 6 h, which includes
checkpoint overhead and so exceeds the instantaneous logged value):

| arm | step/61035 | s/step | ETA (UTC) |
|---|---|---|---|
| H_6G6E | 44,130 | 1.66 | Sep 21 ~23:45 |
| H_6G6S | 40,750 | 1.55 | Sep 22 ~00:40 |
| h400M | 57,030 | 9.55 | Sep 22 ~02:30 |
| sandwich | 33,000 | 2.65 | Sep 22 ~12:30 |

Binding arm is the sandwich at ~Sep 22 12:30Z, ~35 h before the Sep 24 00:00Z deadline. The
pre-compact "~7-8.5 h" h400M estimate was WRONG; corrected 07:10Z from the measured 8.96 s/step.

Two `FAILED to link` lines in `watchdog.log` (abl_I step 122070, abl_H_6G6S step 30600) are
**benign** — the in-trainer mechanism had already created those directories, so the backstop's
`cp` failed with *File exists*. Both arms' anchors verified present, incl.
`abl_I/.../tok32B_step122070_actual32.00B`.

# 17. SESSION 2026-09-21 (evening) — the prefactor is NOT the gradient; seven arms were unwatched

## 17.1 `hopfield_grad_scale` does not return `grad_h E`, and the two gelu knobs are COUPLED

`E = mean_j gelu(Wh)_j^2` (what `energy_per_token` computes), so
`grad_h E = (2/I_e) W^T [gelu(Wh) . gelu'(Wh)]` — the correct prefactor is **2/I_e**.

`_gelu_and_grad` does NOT return the true derivative for the default method: `"sigmoid"` gives
`phi' = 0.5*sigmoid(1.702u)`, HALF the derivative. `"erf_exact"` gives the true one. **So a
prefactor correct under one convention is wrong by 2x under the other.** Verified by autograd on
`energy_per_token` (independent of `forward`) in float64 — ratio `||forward||/||grad_h E||` and
the cosine between them:

| grad_scale | gelu_grad_method | ratio | cos |
|---|---|---|---|
| **`exact` (NEW, 2/I_e)** | **`erf_exact`** | **1.000000** | **1.00000000** |
| `exact` | `tanh_exact` | 1.000467 | 0.99999979 |
| `exact` | `sigmoid` | 0.510497 | 0.99682862 |
| `mean` (4/I_e) | `sigmoid` | 1.020995 | 0.99682862 |
| `mean` | `erf_exact` | 2.000000 | 1.00000000 |
| `sqrt_consistent` | any | sqrt(I_e) | 1 |

**Every shipped hopfield arm therefore takes a step `sqrt(I_e)` = 66.93x the true gradient at
I_e=4480.** A mode `exact` = `2.0/intermediate_size` was ADDED to
`_hopfield_grad_prefactor` + `_HOPFIELD_GRAD_SCALES` + the 3 asserts in `config/mlp.py`.
Additive only; every existing config keeps its behaviour.

## 17.2 Why the knob is hopfield-ONLY — a real structural asymmetry, paper-worthy

* **hopfield** `E = (1/I_e) sum_j gelu(w_j h)^2` — a sum of I_e **positive** terms, O(I_e), so
  1/I_e is forced to keep `E = O(1)`. But the gradient is a sum of I_e **incoherent vectors**,
  only O(sqrt(I_e)), leaving `||grad E|| ~ 2 sigma^2 d / sqrt(I_e)`. **You cannot have
  `E = O(1)` and `||grad E|| = O(1)` at once for a positive-definite energy of this form.**
* **w1w2** `E = -(1/sqrt(I_e)) sum_j phi(W1h)_j (W2h)_j` — a sum of **signed** terms, already
  O(sqrt(I_e)). ONE `1/sqrt(I_e)` inside the ENERGY normalises E and its gradient together, and
  `W1W2FFEnergy.forward` returns `-grad_h E` exactly, with no knob. No tension to resolve.

Call sites are hopfield-only: `HopfieldFFEnergy.forward`, `_HopfieldExpert.forward`,
`BoltzmannMoEFFEnergy._forward_fused` / `_forward_sparse`.

## 17.3 The inflation was installed for a symptom whose cause was found 3 days LATER

`hopfield_grad_scale` added **2026-09-12** (`673af6ea`) to cure an FF branch measured at 0.04% of
`grad_E`. The routing sign inversion was found **2026-09-15** (`6a99703f`), and the follow-up
commit `9b4748ca` is titled *"the inverted sign caused the dead FF branch AND was acting as a load
balancer."* After `e_sign_override: pos` fixed the routing, the 66.93x inflation stayed.

**Correction to the "negative bump" intuition:** minimising `gelu(u)^2` does NOT drive `Wh` into
the bump. `gelu^2 -> 0` both as `u->0` and `u->-inf`; the bump at `u=-0.7518` is a **local
MAXIMUM** with value 0.028890 — a barrier between the two minima. Anti-routing drives `Wh->0`
(rows orthogonal to h) or `Wh->-inf`.

**Magnitudes separate into two mechanisms, only one is large.** Selection alone: at random init
(K=16, I_e=4480, d=768, sigma=0.02) picking argmin instead of argmax energy costs only **1.128x**,
because all K energies are near-equal there (the documented effK 7.999/8). The **operating
point** is what matters: sweeping std(Wh) from 1 -> 0.02 with the exact 2/I_e prefactor, the
gradient falls **~4500x** (3.515e-2 -> 7.815e-6). The arm that motivated the patch sat at
E~0.0126, i.e. std(Wh)~0.2-0.3, ~17x below the O(1) regime, and `||W||_F` fell 70.9 -> 20.5
(a further ~12x in a gradient scaling as ||W||^2). So the inertness is consistent with a
COLLAPSED OPERATING POINT, not with 1/I_e being the wrong normalisation.

**COUNTER-EVIDENCE that must be reported:** the shipped 66.93x arm is NOT unstable.
`cmix_134M_pure_32B_sparse` ran the full 32B with **ZERO grad_norm excursions above 5.0** across
all 12,207 logged steps.

Full writeup pushed to Overleaf as `sec/debug.tex` (commit `71fe232`), `\input` at the end of the
appendix, marked for removal before submission.

## 17.4 The ladder: abl_J / abl_K / abl_L (RUNNING)

All off `configs/cmix/cmix_134M_pure_32B_sparse.yml` — which IS the `sqrt_consistent` arm and is
already DONE at 32B, so only 3 new arms were needed. Same seed (42, default, set in no config),
same data order, iso-token 262,144 tok/step, **identical TOTAL/ACTIVE/FLOPwt** (the knob is a
parameter-free scalar). Parent is `layer_iterations: [12]` = `1x12E`, NOT 1x6E.

| arm | grad_scale | gelu | x grad_h E | job |
|---|---|---|---|---|
| `cmix_134M_pure_32B_sparse` | `sqrt_consistent` | sigmoid | 66.93 | DONE 32B: **Avg11 42.00 / MMLU 24.43**; 8B anchor **39.93 / 24.84** |
| `abl_K_..._gsInvSqrt_1gpu` | `inv_sqrt` | sigmoid | 16.73 | 1830288 |
| `abl_J_..._gsMean_1gpu` | `mean` | sigmoid | 1.000 (cos 0.997) | 1830287 |
| `abl_L_..._gsExact_1gpu` | `exact` | `erf_exact` | **1.000 (cos 1.000)** | 1829569 |

Separates **magnitude** (1x / 16.7x / 66.9x) from **direction fidelity at fixed magnitude**
(J cos 0.9968 vs L cos 1.0). Reference loss to compare against: 3.4180 @ 4.00B, 3.3344 @ 8.00B.
Milestones set to `[1, 2, 4, 8, 16, 24, 32]` so 1B (step 4000 = 1.049B) is benchmarkable.
Compare with `scripts/compare_prefactor_ladder.sh`.

**DO NOT read the ordering before step 2000.** At steps 300-400 K (16.7x) led L by 0.90 nats and
L led J by 0.06 — inside LR warmup, where early descent speed does not predict final quality.
User instruction 2026-09-21: wait.

## 17.5 1 GPU is the ONLY obtainable shape, and mbs CANNOT be raised there

Probed empirically (all PENDed except the last): 8 GPU/1 host PEND; 2 GPU/1 host PEND;
**8 hosts x 1 GPU in one job (`span[ptile=1]`) PEND** — *"requirements for reserving resource
(ngpus_physical) not satisfied: 212 hosts"*, because LSF must reserve on all 8 hosts at once;
standalone **1 GPU RUNS immediately**. So scattering does not work: separately-scheduled 1-GPU
jobs cannot form one NCCL world.

`bhosts -gpu` **overcounts free GPUs 4x** — it ignores host STATUS. Raw count said 1038 free;
filtered to `STATUS=ok` it is **256**, with 632 of the "free" GPUs on `closed` hosts and 112 on
`unavail`. And 23 of the 32 fully-free ok hosts carry
`ADMIN ACTION COMMENT: "move to qecjobs reservation"` (grp_qec, window to 2038). Our `normal`
fairshare PRIORITY is **0.001** vs 0.333 for idle users.

**mbs at 1 GPU:** iso-token variants (mbs x ga held at 64) — mbs 2/ga 32 **RUNS**; mbs 4/ga 16
**OOM** (77.20 of 79.18 GiB, died allocating 2.19 GiB); mbs 8/ga 8 **OOM**. Cause is the absence
of FSDP sharding. **So the S12.9 sparse-speedup regime (16,384 tok/call, 2.1-2.4x) is
UNREACHABLE at 1 GPU** — it needs the multi-GPU config. Worth testing mbs 4 if the ladder is
rerun at 8 GPUs.

**Sparse IS a 2x win at this shape**, contrary to the S12.9 worry about 8,192 tok/call:
step time 6.61 -> **3.26 s/step** across `sparse_start_step: 300` (7.93 s at the recompile step).
**Never read a step_time before step 300** — the pre-300 dense phase evaluates all K=16 experts.

## 17.6 ALL SEVEN LIVE ARMS WERE UNWATCHED — fixed

None of the four grp_ebm arms nor the three ladder arms was in `watchdog_jobs.conf`; any
preemption, node fault or disk-full death was permanent until a human noticed. All seven added
(backup: `watchdog_jobs.conf.bak-20260921`). Safe while running because of the **name-based
check (2026-09-14)**: the watchdog logged all seven as *"already in queue as jid=N (RUN);
adopting, NOT resubmitting"*, and `bjobs` confirms exactly one job per name. The conf is
re-read every cycle (`done < "$CONF"` sits inside the outer loop), so no restart was needed.
NOTE `/u/ndehmamy/Code/...` and `/proj/dmfexp/nima/Code/...` are the SAME FILE (inode
277994610529) — the watchdog logs the former path.

## 17.7 New unattended machinery (all self-resubmitting, survive session death)

| job | name | what |
|---|---|---|
| 1825457 | `boltz_eval_finish` | loops `auto_eval_on_finish.sh` every 300 s. **Nothing unattended covered cmix/iclr_26 arms before this** — `boltz_auto_eval` drives `collect_flops_wave_20260912.sh`, whose CFGDIRS are the eight `configs/iclr_*` dirs and include NEITHER `configs/cmix` NOR `configs/iclr_26`. Queue deliberately preemptable; do NOT set EVAL_QUEUE=ebm. |
| 1830299 | `boltz_ladder_handoff` | promotes ladder arms 1 -> 8 GPUs as grp_ebm frees. Submits the 8-GPU job, polls 15 s until RUN, THEN kills the 1-GPU twin (killing first risks losing the freed GPUs). **Deregisters the `*_1gpu` watchdog entry before killing**, else the watchdog resubmits a twin racing the same save_path (rule 5). |

## 17.8 Two finished runs had NEVER been evaluated — now scored (free datapoint)

`results/tok90k` is NOT the broken wave; it is **the FIX for it** (`slope90k_*` were the broken
ones, and they total only ~13 G). I nearly deleted 272 G of completed corrected-schedule results.
Both complete arms reached 90,000/90,000 = **23.59B tokens** (262,121 / 262,106 tok/step measured
empirically, `DeviceMesh(ddp=4)` = 4 GPUs x mbs 4 x ga 4 x 4096), on the **OLD 100%-web datamix**:

| arm | Avg11 | MMLU |
|---|---|---|
| `t90k_switch_lastisoP` (Switch) | **44.93** | 25.40 |
| `t90k_hybrid_K32top2` (energy) | 44.42 | 24.53 |
| **Δ (energy − Switch)** | **−0.51** | −0.87 |

Direction agrees with §14.1 at 134M/iso-FLOP (Switch 45.43 vs hybrid 44.82, **−0.61**) — two
budgets, two schedules, within 0.1pp. But **−0.51 is inside the seed noise floor** (colleagues'
matched-seed Avg11 |Δ| = 0.32pp at 400M), and the datamix makes these comparable ONLY to each
other. Report parity-to-slightly-behind.

## 17.9 Disk and storage — we are NOT the problem; cleanup is not the lever

Volume hit **99% (5.2 T free, from 8.1 T that morning)**; burn measured **1.15-1.34 TB/h
sustained**, later easing to ~0.35. **Our whole footprint is 1.4 T of 345 T used = 0.4%**, below
the ~3.4 T average across ~100 user dirs, and the live arms are size-stable (`max_to_keep: 2`).
704 GB is reclaimable from our tree without losing a result (sharded `global_step*` of runs that
are finished AND unsharded AND evaluated; biggest: `iclr_scale/scale32B_boltz_sinkhorn` 200 GB,
`cmix/cmix_134M_gptswitch` 180 GB) — **but at 1.3 TB/h that buys 19 minutes, so it is hygiene,
not mitigation. User declined; nothing was deleted.**

`iclr_scale` is on the **old 100%-web mix** (0.5/0.5, `split: 99.5,0.5,0`, `.cache/megatron`),
NOT cmix — so not comparable to any cmix arm, same confound as tok90k.

**Per-user storage ranking is NOT obtainable without admin tools.** `du` over 350 T ran 2 h with
no output (killed — heavy metadata load while 7 arms checkpoint); `mmlsfileset` says *"No filesets
found owned by this userId"*; `mmrepquota` is Permission denied. Ask an admin for `mmrepquota` or
an `mmapplypolicy` LIST report.

Remaining space we actually need to the deadline is only **~150 G** (400M ckpt 4.5 G, 134M ~2 G;
remaining milestone anchors are hard links and DO hold data once the original is pruned).
A 150 G `fallocate` ballast + auto-release watcher was offered and is undecided.

## 17.10 FIRST TRUSTWORTHY LADDER READING — the 1B anchor (2026-09-21 23:34Z)

`lm_loss` in nats, LOWER better. Iso-token 262,144 tok/step, same seed (42, default), same data
order, identical TOTAL/ACTIVE/FLOPwt. Only the FF-branch prefactor differs.

| step | tokens | ref `sqrt_consistent` 66.9x | `abl_K` 16.7x | `abl_L` **exact 1.0x** | `abl_J` mean 1.0x |
|---|---|---|---|---|---|
| 2,000 | 0.52B | **4.3486** | 4.3733 | 4.8171 | 4.8187 |
| 2,500 | 0.66B | **4.1659** | 4.1840 | 4.5538 | 4.5379 |
| 3,000 | 0.79B | **4.0459** | — | 4.4274 | 4.3982 |
| 4,000 | **1.05B** | **3.8183** | — | **4.1699** | — |

**Exact energy descent is 0.35 nats behind the shipped arm at 1B.** Ordering is monotone in
prefactor magnitude (66.9x < 16.7x << 1.0x). Three findings:

1. **The effect is MAGNITUDE, not DIRECTION.** J (cos 0.9968) and L (cos 1.0000) are
   indistinguishable — the sigmoid `gelu'` approximation costs nothing measurable.
2. **The benefit SATURATES well below sqrt(I_e).** K at 16.7x nearly matches the reference at
   66.9x: 4.3733 vs 4.3486 (step 2000), 4.1840 vs 4.1659 (2500) — gaps of 0.025 and 0.018 nats.
   So most of the gain is already bought at 16.7x, and 66.9x adds little.
3. **A ~1.45x horizontal STEP-SHIFT, stable over a 2-nat span.** Asking at which step K first
   reaches each 1x arm's loss: L/K = 1.49, 1.38, 1.38, 1.51, 1.49 and J/K = 1.63, 1.48, 1.44,
   1.53, 1.46 across losses 6.5 -> 4.6. From step 600 onward all three descend at the SAME rate
   (0.697 / 0.709 / 0.696 nats per 400 steps), so the deficit was acquired entirely in a plateau
   over steps 200-300 (J +0.095, L +0.148 vs K +0.590) and never repaid. That plateau is the
   DEAD-BRANCH REGIME reproducing on demand: at 1x the branch returns the true grad_h E, which
   is ~67x smaller than what the shipped arm applies.

**TWO CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS.**
* The LR decay is ~120,000 steps away (2000 warmup + 0 constant + 120,070 cosine to 0.1x).
  Parallel mid-schedule curves often CONVERGE at the floor, so a 0.35-nat gap at 1B may be much
  smaller in final Avg11.
* **K's data order is CONTAMINATED.** Preempted 4x on the preemptable queue, it replayed steps
  2000-2950 two-to-three times (rollbacks to ck=2000). Small on 122,070 steps but it is the one
  arm whose data order is no longer identical to the others'.

**Current reading:** exact descent costs ~0.35 nats at 1B / ~1.45x in tokens, so the 1/sqrt(I_e)
inflation EARNS ITS PLACE even though it was installed for the wrong stated reason (§17.3). This
REVERSES the direction the original hypothesis pointed — it is not an artifact to remove but a
deliberate, now-quantified departure from exact descent. The paper can state it either way, but
must not claim the branch is doing energy descent.

Milestone anchors so far: L `tok1B_step4000_actual1.05B`. J and K not yet at 1B (K is behind from
preemption). Re-run with `scripts/compare_prefactor_ladder.sh`.

## 17.11 ⛔ A 1-GPU CHECKPOINT CANNOT BE LOADED AT 8 GPUs — the promotion design was invalid

**2026-09-22 00:17Z.** `ladder_gpu_handoff.sh` promoted `abl_K_gsInvSqrt` from 1 GPU to 8,
correctly deregistered the `_1gpu` watchdog entry and killed the 1-GPU twin — and the 8-GPU job
then died in under a minute:

```
FileNotFoundError: .../abl_K_134M_pure_1x12E_gsInvSqrt/global_step2000/lr_scheduler/lr_scheduler-4.pt
```

A checkpoint written at `world_size=1` contains ONE FILE PER RANK and only rank 0:

| subdir | contents | reshards across world sizes? |
|---|---|---|
| `model/` | `__0_0.distcp` | **YES** — torch DCP |
| `optimizer/` | `__0_0.distcp` | **YES** — torch DCP |
| `lr_scheduler/` | `lr_scheduler-0.pt` (n=1) | **NO** — plain per-rank, loaded by rank index |
| `rng_state/` | `rng_state-0.pt` (n=1) | **NO** — plain per-rank |

Model and optimizer would have resharded fine; `lr_scheduler` and `rng_state` are plain
per-rank `.pt` files and an 8-GPU job demands ranks 0..7. **So changing GPU COUNT on a resume is
impossible with this checkpoint format, no matter how the token budget is held constant.** The
whole handoff rested on "iso-token via `mbs x ga` makes the checkpoint portable" — iso-token it
is, portable it is NOT.

**Consequence and the near-miss.** The arm was left with nothing running AND deregistered from the
watchdog, i.e. unprotected. Worse, the watcher had already started the same promotion on `abl_L`
(8-GPU job 1835383) two minutes later; had it completed, `abl_L` would have lost a healthy 4,900-step
run the same way. The watcher was killed mid-wait and 1835383 force-killed. `abl_K` was restored by
re-registering its conf entry and resubmitting AT 1 GPU (job 1835411), resuming from ck=2000.

**RULES.**
1. **NEVER change GPU count across a resume** for these runs. A `_4gpu`/`_8gpu` sibling config is
   only valid for a run STARTED at that width, never for continuing one started at another.
2The `*_1gpu` configs are fine — but an arm started at 1 GPU finishes at 1 GPU or restarts from
   scratch.
3. `ladder_gpu_handoff.sh` is DEAD and must not be resubmitted. Its header now says so.
4. A completion-only state file cannot tell you about IN-FLIGHT work: the first restart of this
   watcher stranded an 8-GPU job for `abl_J` (1833574) because the state file was empty while a
   promotion sat in its 20-minute confirm wait. Read the watcher's LOG, not just its state.

## 17.12 The `pure` deficit looks CAPACITY-limited, and 12x1E has never been built

At 7.86B, same backbone, only the FFN block differing (Avg11 higher better, wiki ppl lower better):

| arm | Avg11 | ppl |
|---|---|---|
| hybrid dense K16 | **44.78** | 40.73 |
| hybrid `6G+1x6E` top-2 | 43.91 | 40.49 |
| **pure `1x8E`** | **40.21** | **69.66** |

and at 32B `pure 1x12E` is 42.00 / ppl 66.08 against the hybrid's ~40.7 ppl. **The perplexity is
60-70% worse**, far too large for a routing detail, and pure gained only 40.21 -> 42.00 from
7.86B to 32B while ppl stayed ~67-70 — the signature of a CAPACITY limit, not undertraining.
Structural cause: pure holds 55.1M in ONE block applied 12 times, so all 12 depth positions share
one weight set.

**No config anywhere has >=4 NON-recurrent energy-MoE blocks at 134M**, so `12x1E` is untested.
Nearest existing evidence: `abl_H_400M_6G6E_deep` (6 GPT + 6 DISTINCT energy blocks, no
recurrence) finished 32B at 2026-09-22 00:01Z and is being evaluated — that is the first real
datapoint on distinct-blocks-vs-recurrence. `bharat_1B_18L_allMoE_boltz` (18 distinct blocks) is
the same idea at 1B, never run.

**Proposed ablations, cheapest-first.** (a) `3x4E` or `4x3E` at 134M — same 12 applications, 3-4
DISTINCT blocks; isolates weight sharing from recurrence and is the missing middle. ~15 h on
8 GPUs. (b) `12x1E` is only informative at iso-WIDTH: at iso-total-param it forces
I_total = 71680/12 = 5973, i.e. 16 experts x 373, straight into the documented B-series
tiny-expert failure. (c) Give pure `softmax_attention` — pure uses `energy_attention` at all 12
positions while the hybrid's 6 GPT layers use softmax, a second unisolated confound.
(d) **There is NO 12G dense baseline at 134M** — the dense arms are `abl_D`/`abl_F` at SIX layers.
A 12-layer dense GPT would cost ~7 h and is the missing target for "close the gap to 12G".

`cmix_400M_pure_it4_sparse` (`1x4E`, I_e=17920) is NOT a usable datapoint: abandoned at
1,400/61,035 steps (0.37B tokens), no eval.

## 17.13 DISK CLEANUP 2026-09-22 03:05Z — sharded checkpoints deleted, what is no longer re-evaluable

The volume hit **554 G free of 350 T** with bursts up to 4.6 TB/h (all external — our tree was
1.4 T, 0.4% of the 349 T used). User approved reclaiming redundant SHARDED checkpoints from runs
that are finished AND unsharded AND already evaluated. **554 G -> 866 G free (+312 G net);** our
tree 1.4 T -> 1,019 G.

| target | deleted | KEPT |
|---|---|---|
| `iclr_scale/scale32B_boltz_sinkhorn` | all 40 `global_step*` (179 G -> 1.5 G) | `unsharded`, `unsharded_mucal`, 2 eval JSONs |
| `cmix/cmix_134M_gptswitch` | 88 of 90 `global_step*` (137 G -> 4.1 G) | `global_step31000`, `global_step62000`, `unsharded_step30000`, `unsharded_step90000`, **6/6 evals, 6/6 milestones** |
| `tok90k/t90k_pure_T12` | the whole arm (47 G) | nothing — it had no unsharded and no evals |

**WHAT THIS COSTS: those two runs can no longer be re-evaluated at an ARBITRARY step.** Only the
checkpoints listed under KEPT survive. For `cmix_134M_gptswitch` that is steps 30000 (the
iso-token rescue), 31000 and 62000 (milestone-anchored) and 90000 — nothing else.
`scale32B_boltz_sinkhorn` has only its two unsharded snapshots. If a future comparison needs a
different intermediate budget from either arm, it must be RETRAINED.

**Two mechanical lessons.**
1. `global_step*` dirs that are HARD-LINKED from `milestones/` free NOTHING when deleted — the
   milestone keeps the inodes alive (check with `stat -c %h` on a `.distcp`; link count 2 means
   anchored). `gptswitch`'s step31000/step62000 were exactly this, so they were kept deliberately.
2. **`df` LAGS on this GPFS.** The first post-deletion `df` showed +0 G and nearly got reported as
   a failed deletion; `du` on the target showed 179 G -> 1.5 G immediately and `df` caught up
   ~1 min later. Verify a deletion with `du` on the target, never with `df`.

**Also corrected here:** `t90k_pure_T12` was described in conversation as "abandoned at
1,400/90,000" — that was a conflation with `cmix_400M_pure_it4_sparse`. It was at
**31,000/90,000 (8.1B tokens)**, no unsharded, no evals, old 100%-web datamix.

## 17.14 ⚠ `df` ON THIS CLUSTER REPORTS A **FILESET QUOTA**, NOT FILESYSTEM CAPACITY

**2026-09-22 03:15Z, after the admins said "disk is not full, main GPFS is 70%".** Both were true:

| path | what it is | size | used | avail | use% |
|---|---|---|---|---|---|
| `/gpfs/ess6000-1` | **the FILESYSTEM** | **7.3 P** | 5.1 P | **2.2 P** | **70%** |
| `/proj/dmfexp` | fileset QUOTA | 350 T | 350 T | 832 G | **100%** |
| `/proj/datasets` | fileset QUOTA | 600 T | 593 T | 7.1 T | 99% |
| `/u/ndehmamy` | fileset QUOTA | 100 T | 23 T | 78 T | 23% |

**The tell was visible for hours and was missed:** `df` prints the SAME device name `ess6000-1`
with a DIFFERENT size per path (350 T for `/proj/dmfexp`, 100 T for `/u/ndehmamy`). Same device +
different size = per-fileset quota reporting, NOT separate filesystems. This session recorded
`$HOME` as "a separate filesystem at 23%" (§17.9) — it is the same filesystem, a different fileset.

**What this does and does not change.**
* The WRITE RISK WAS REAL. A quota boundary stops writes exactly as hard as a full disk, so
  `sandwich` would still have failed its checkpoints at 0 G of quota.
* The framing "the filesystem is filling, ~3 h to zero" was WRONG. The storage was never going
  to be exhausted; our ALLOCATION was.
* **The correct fix is a quota increase, not deletion or ballast:** ask an admin to raise the
  `dmfexp` fileset quota on `ess6000-1` — it is at 350 T/350 T while the filesystem has 2.2 P
  free. `mmsetquota`, seconds of work, no data destroyed. The ~390 G of sharded checkpoints still
  reclaimable in our tree can stay.

**RULE: before treating a `df` percentage as a capacity emergency, check whether the path is a
fileset.** `df -h /gpfs/<device>` gives the real filesystem; `df -h <your path>` gives your quota.

## 17.15 ⛔ POST-HOC mu CALIBRATION MAKES THESE ARMS WORSE — keep the mu=0 evals

**2026-09-22 04:15Z.** An agent flagged that `sinkhorn_persist_mu` is absent from the 134M `cmix_*`
arms (it defaults to **False**, `config/mlp.py:406`), so they train with a mu-tilted router and
EVALUATE at mu = 0 — half of pre-flight rule 9. True. It was tested on all three completed 32B
arms with `calibrate_sinkhorn_mu_20260915.py` (64 batches, bs 2, seq 4096). **Calibration made
every one of them WORSE:**

| arm | iters | Avg11 before (mu=0) | after mucal | Δ | ppl before | after |
|---|---|---|---|---|---|---|
| `cmix_134M_pure_32B_sparse` | **12** | **42.00** | 40.98 | **−1.02** | 66.08 | 67.36 |
| `cmix_134M_hybrid_32B_sparse` | 6 | **44.82** | 44.50 | −0.32 | 41.06 | 41.08 |
| `cmix_134M_sandwich_32B_sparse` | 6 | **43.45** | 43.37 | −0.08 | 40.98 | 40.99 |

**USE THE `unsharded_step122070` NUMBERS. Do not use `unsharded_mucal`.** The calibrated dirs are
left on disk for the record only.

**Why it hurts.** The post-hoc mu is estimated from 64 batches of ONE prefix
(`web-nemotron-cc-hq-p2_0`) against a training EMA that converged over 122,070 steps on the cmix
70/30 web/math blend. A mis-estimated mu is worse than none, and the damage SCALES WITH ITERATION
COUNT (12 iters −1.02, 6 iters −0.32/−0.08) — consistent with a wrong per-iteration dual
compounding through the stack. Note the script DOES now build a per-iteration buffer
(`probed 12 mu solve(s) per forward -> sizing the buffer (n_iter, K)`, counts `[64]*12`), so its
docstring's "KNOWN LIMITATION: only ONE sinkhorn_mu buffer" is OUT OF DATE — the collapse is not
the problem.

**RULE 9's MAGNITUDE IS CONFOUNDED AND SHOULD BE RESTATED.** Its "+1.587 nats at 8 iterations,
+1.827 at 12" traces to the calibration script's header, which compares an arm using
`load_balance_bias` (a persistent buffer that DOES reach eval) at Avg11 **41.49** against *its
Sinkhorn twin* whose mu was dropped at **36.16**. Those are TWO DIFFERENT BALANCING MECHANISMS on
TWO DIFFERENT ARMS — not one checkpoint with and without mu. The clean within-arm test is the
table above, and it has the OPPOSITE SIGN. Our arms set `balance_rate` nowhere and their
checkpoints carry NO routing-state keys at all, so there is no bias to compare against.

**CONSEQUENCE FOR §17.12: THE CAPACITY-LIMIT DIAGNOSIS OF `pure` STANDS.** Its 42.00 / ppl 66.08 is
NOT an evaluation artifact. `pure` really is the weaker model, its perplexity really is ~60% worse
than the hybrids', and the proposed `3x4E` / `6E` / 12-layer-dense work IS aimed at a real deficit.
The hypothesis that it was a rule-9 artifact was raised here and is REJECTED by measurement.

**AND THE 400M RESULTS WERE NEVER AT RISK.** `abl_H_400M_6G6E_deep`, `cmix_400M_hybrid_sparse` and
`cmix_400M_sandwich_sparse` all set `sinkhorn_persist_mu: true`; the Switch arms have no Sinkhorn.
So 47.29 / 48.24 / 46.92 / 48.83 and both iso-FLOP pairs stand as measured.

# 18. THE 400M TIER IS THE EVIDENTIAL ONE, AND IT SAYS THE LEARNED GATE WINS (2026-09-22)

## 18.1 The complete 400M four-arm table — two clean iso-FLOP pairs

All 400M total, 32.0B tokens, `configs/cmix` datamix. Avg11 = 11-task mean x100 (higher better);
MMLU and GSM8K-CoT separate accuracy % ; FLOPwt = M parameter-applications per token (recurrence
MULTIPLIES it, so it is the compute measure, not `active`).

| arm | Avg11 | MMLU | GSM8K | FLOPwt | total |
|---|---|---|---|---|---|
| `6G6S` Switch, **distinct** | **48.83** | **27.10** | 3.26 | **239.5M** | 399.9M |
| `6G1x6S` Switch, recurrent | 48.24 | 26.74 | 3.26 | 301.0M | 400.0M |
| `6G1S` Switch, unmatched | 47.44 | 26.55 | 2.73 | 231.7M | 400.0M |
| `6G1x6E` energy, recurrent | 47.29 | 25.72 | 2.12 | 299.4M | 399.8M |
| `6G6E` energy, distinct | 46.92 | 25.18 | 2.12 | 238.0M | 399.9M |

**TWO ISO-FLOP PAIRS, BOTH FAVOURING THE LEARNED GATE:**
* recurrent `6G1x6E` vs `6G1x6S`: **-0.95pp** (FLOPwt within 0.5%, active within 0.2%)
* distinct `6G6E` vs `6G6S`: **-1.91pp** (FLOPwt within 0.6%, total within 0.002%)

**THE GAP WIDENS WITH SCALE.** 134M iso-FLOP -0.61pp; `t90k` 23.59B -0.51pp; 400M recurrent
-0.95pp; 400M distinct **-1.91pp**. The cleanest comparison — simultaneously iso-FLOP, iso-param,
iso-structure and at the evidential scale — shows the LARGEST deficit, and -1.91pp is far outside
the 0.32pp matched-seed noise floor. This cannot be written off as noise the way the sub-1pp 134M
results could.

## 18.2 RECURRENCE HELPS ENERGY AND HURTS SWITCH — the one asymmetry that favours the energy story

| | distinct (6 blocks x1) | recurrent (1 block x6) | delta from recurrence |
|---|---|---|---|
| energy | 46.92 | 47.29 | **+0.37pp** |
| Switch | **48.83** | 48.24 | **-0.59pp** |

Mechanistically sensible: iterating ONE block is descending an energy landscape repeatedly, which
is what the energy formulation is for, whereas a learned gate gains nothing from reapplying
identical weights. **But the exchange rate is poor** — energy buys +0.37pp for +26% FLOPwt
(238.0 -> 299.4M). And the best arm overall, `6G6S`, wins on BOTH axes: highest Avg11 at the
LOWEST FLOPwt of the four.

## 18.3 DENSE HAS BEATEN EVERY MoE VARIANT AT 134M

134M, 32B, cmix. Answering "did the dense models ever perform better": yes, on both matching
conventions.

| arm | Avg11 | FLOPwt | kind |
|---|---|---|---|
| `6G+1x6E` base EGPT, **NO MoE** | **45.87** | 141.7M | recurrent energy, single FFN |
| `6G` dense iso-total | 45.55 | 134.3M | dense |
| `6G1x6S` Switch | 45.43 | 143.2M | MoE |
| `6G` dense iso-active | 45.01 | **123.3M** | dense |
| gptswitch | 44.87 | 123.5M | MoE |
| `6G1x6E` energy MoE | 44.82 | 141.7M | MoE |
| pure `1x12E` | 42.00 | 185.1M | MoE |

**Dense iso-active beats the energy MoE while spending 13% FEWER FLOPs** (45.01 @ 123.3M vs
44.82 @ 141.7M). **The best arm in the tier has no mixture at all.** So the recurrent ENERGY layer
contributes (+0.32pp over dense at ~5% more FLOPs); adding the MIXTURE costs 1.05pp (45.87 ->
44.82). CAVEAT: the top four span 0.86pp against a 0.32pp seed floor, so their internal ordering
is not resolvable on one seed; `pure`'s 42.00 is well outside noise and is real (§17.15).

**WHY `abl_H_400M_6G6G_deep_isoactive` MATTERS AND IS NOW RUNNING.** It is 12 DENSE layers at
238.0M FLOPwt — iso-FLOP with BOTH 400M MoE arms while carrying only 238M total params, 40% fewer
than their 400M. It had **stalled at 16,200/61,035 (26.5%) with no job and no watchdog entry**;
resumed 2026-09-22 04:18Z as job 1840680 and REGISTERED in `watchdog_jobs.conf`. If it lands near
48.8 then the MoE's extra 162M parameters buy nothing at 400M either, which is a sharper claim
than "Switch beats energy".

## 18.4 Temperature sweep (running) — and learnable temperature is NOT implemented

`temperature` is `float(temperature)`, a plain Python float used at 7 sites in `energy_ff.py`.
There is **no learnable-temperature machinery anywhere in the tree**, and no `energy_scale_mode`
either — so the colleagues' config header claiming `sqrt_inv_d` "instead of a learnable
temperature" refers to something that does not exist here.

Fixed-tau sweep first, to find out whether tau matters at all before implementing it. Base
`cmix_134M_hybrid_32B_sparse`, ONE field changed (verified: 3-line diff, datamix identical),
`num_training_steps` deliberately left at 122,070 so the LR schedule matches the control, arms
STOPPED at step 32,000 = 8.39B where the control has an anchor:

| tau | Avg11 @ 8B | source |
|---|---|---|
| 0.35 | pending | `abl_T_134M_hyb_tau0p35`, ETA 06:31Z |
| **1.0** | **43.59** (MMLU 23.48) | control's own `unsharded_tok8B_step32000`, evaluated 05:13Z |
| 2.0 | pending | `abl_T_134M_hyb_tau2p0`, starts when 0.35 frees its GPUs |

**Yardstick for reading it:** the control gains +1.23pp Avg11 (43.59 -> 44.82) from 8B to 32B, so
a tau effect under ~1pp at 8B is comparable to what a quarter of the budget buys.

**PRE-REGISTERED PREDICTION:** with `routing_norm: zscore` the logits are already per-token
standardised, which removes exactly the scale tau controls — so I expect the three points to be
FLAT, and if they are, a learnable tau is not worth implementing for these configs. A learned tau
is more interesting with `routing_norm: none` or `sqrt_width`.

**If it IS implemented later, three decisions and one hazard.** Parameterise `log_tau` (so
`tau = exp(log_tau)` stays positive with no clamp and takes multiplicative steps); decide
per-block vs per-iteration (a recurrent block shares one tau across 6-12 applications, the same
collapse the mu buffer had to fix with an `(n_iter, K)` shape); and note the HAZARD that
`temperature` ALSO multiplies the reported free energy at `energy_ff.py:1273,1526,1711,1932`, so
making it learnable makes `energy_per_token` a function of a trained quantity — and
`energy_descent_loss_coef` / `energy_action_loss_coef` assume `ffwd_out == grad_h E`, safe only
because both are 0 everywhere.

## 18.5 Table 6 (`tab:status`) regenerated and pushed — Overleaf `4b9ef77`

23 data rows, 11 columns. New/updated: 400M hybrid complete at 47.29; `6G6S` 48.83 and `6G6E`
46.92 added; 12G dense 27% (no job) -> 31% (RUN); sandwich 34% -> 82%.

**A NEAR-MISS WORTH REMEMBERING.** The replacement was anchored on the `% GENERATED` comment, but
`\resizebox{\textwidth}{!}{%` sits on the line ABOVE that marker, so the line-range edit DELETED
it — leaving an unmatched `}` (a compile error on Overleaf, undetectable locally since `pdflatex`
cannot run here) and an 11-column table free to overflow the margin. The brace-balance check
caught it; a check for "are the new numbers present" would NOT have. **Always verify brace balance
and environment counts after a programmatic .tex edit, not just content.**

DELIBERATELY EXCLUDED from Table 6: the `t90k` pair (old 100%-web datamix, would falsify the
caption's "same data mixture" claim — they live in `sec/debug.tex`), and all three `unsharded_mucal`
numbers (worse than mu=0, see §17.15).

## 18.6 TEMPERATURE SWEEP: tau is inert, and Avg11 at 8B is NOISIER THAN THE LOSS

Three arms, ONE field changed from `cmix_134M_hybrid_32B_sparse` (verified 3-line diff, datamix
identical), `num_training_steps` left at 122,070 so the LR schedule matches, all STOPPED at step
**32,000 = 8.39B** where the control has an anchor. tau=1.0 is the control's own
`unsharded_tok8B_step32000`, so it cost nothing.

| tau | Avg11 | MMLU | wiki ppl | lm_loss @32k |
|---|---|---|---|---|
| 0.35 | 42.87 | **26.53** | **56.78** | **2.9250** |
| **1.00** | **43.59** | 23.48 | 56.84 | 2.9263 |
| 2.00 | 42.20 | 24.34 | 57.35 | 2.9346 |

lm_loss at matched steps (10k / 20k / 30k / 32k):
* tau=0.35 — 3.0144 / 2.9720 / 2.8757 / 2.9250
* tau=1.00 — 3.0180 / 2.9783 / 2.8811 / 2.9263
* tau=2.00 — 3.0192 / 2.9854 / 2.8878 / 2.9346

**LOSS AND PPL ARE MONOTONE AND FLAT.** Lower tau is better at all four checkpoints, so the
gradient is real, but the whole 5.7x range spans **0.0096 nats** and 0.57 ppl. I had predicted
"flat" before running it; "shallow but monotone" is the accurate description.

**THE Avg11/MMLU SPREADS ARE NOISE, AND THIS IS THE MORE USEFUL FINDING.** Avg11 spans 1.39pp and
MMLU 3.05pp, and NEITHER follows the loss ordering (tau=1.0 wins Avg11, tau=0.35 wins loss AND
MMLU AND ppl). A 1.39pp Avg11 difference cannot coexist with a 0.0096-nat loss difference: the
same arm gains +1.23pp Avg11 from 8B (43.59) to 32B (44.82), and that comes with a far larger
loss move. MMLU at 23-27% straddles its 25% chance level and carries no signal at this budget.
**RULE: at 8B/134M, do not trust an Avg11 gap below ~1.5pp unless the loss moves with it.** The
0.32pp matched-seed floor was measured at 400M on 32B arms and is TOO OPTIMISTIC for this protocol.

**VERDICT: do NOT implement a learnable temperature.** There is no gradient worth learning, and
the implementation carries the hazard in 18.4 (`temperature` also multiplies the reported free
energy at `energy_ff.py:1273,1526,1711,1932`, so making it a parameter makes `energy_per_token` a
function of a trained quantity and couples to `energy_descent_loss_coef`/`energy_action_loss_coef`).

**WHY tau IS INERT — and what to ablate instead.** `routing_norm: zscore` standardises the logits
per token, removing exactly the scale tau controls. So the thing worth testing is `routing_norm`
itself. `abl_R_134M_hyb_rnorm_none` (one field, `zscore` -> `none`, same 8B/32,000 protocol) is
RUNNING as job 1843208 at 8 GPUs, 262,144 tok/step. PREDICTION ON RECORD: CLAUDE.md documents that
with `none` the energy magnitude is ~0.0126 against tau=1, giving `effective_n_experts 7.999/8`
i.e. near-uniform routing, so this arm should be CLEARLY WORSE; if it is comparable instead, the
router contributes little either way and that is the more interesting result.

**OPERATIONAL NOTE — REGISTERING AN ARM IN `watchdog_jobs.conf` LAUNCHES IT.** `tau=2.0` was
registered but never submitted; the watchdog saw "not alive, step 0 < target" and started it on
its own at 04:55Z (job 1841480). Useful here, but the conf is a LAUNCHER, not just a safety net.
Submit FIRST, then register, so the watchdog adopts rather than duplicates.

**UNSHARDING A MILESTONE ANCHOR.** `unshard_once.sh LOAD_PATH ITER OUT` needs
`LOAD_PATH/global_step<ITER>`. A milestone dir IS the checkpoint (it holds `model/`, `optimizer/`,
`training_config.yml` directly), so passing it as LOAD_PATH fails. If the live `global_step<ITER>`
has been pruned (tau=2.0 overshot to 40,000, so `max_to_keep: 2` kept 38,000/40,000), make a side
dir with a symlink -- `<sp>/_unshard_src/global_step32000 -> ../milestones/tok8B_step32000_actual8.39B`
-- and pass `_unshard_src`. Keep it OUT of `<sp>/global_step*` so pruning and the milestone scanner
cannot see it.

## 18.7 `routing_norm: none` — the penalty is 0.019 nats, and MY PREDICTION WAS WRONG TWICE

`abl_R_134M_hyb_rnorm_none` = `cmix_134M_hybrid_32B_sparse` with ONE field changed
(`routing_norm: zscore -> none`; verified 3-line diff, datamix identical), same schedule, stopped
at step **32,000 = 8.39B**, 8 GPUs, 262,144 tok/step verified empirically.

| step | `none` | `zscore` (control) | delta |
|---|---|---|---|
| 10,000 | 3.0262 | 3.0180 | +0.0082 |
| 20,000 | 2.9931 | 2.9783 | +0.0148 |
| 30,000 | 2.8939 | 2.8811 | +0.0128 |
| 32,000 | 2.9448 | 2.9263 | **+0.0185** |

**I PREDICTED `none` WOULD BE "CLEARLY WORSE" WITH NEAR-UNIFORM ROUTING (effK 7.999/8). BOTH
HALVES WERE WRONG.** Routing health at step 32,000, K=16 (uniform share 0.0625, max effK 16):

| metric | `none` | `zscore` |
|---|---|---|
| load_effective_n_experts | **15.70** | 15.58 |
| load_max_share | 0.0912 | 0.1039 |
| load_min_share | 0.0374 | 0.0449 |
| **load_mean_token_entropy** | **0.5891** | **0.4694** |

1. **Routing does NOT collapse.** Both arms sit at ~15.6-15.7 of 16 effective experts on LOAD.
   CLAUDE.md's "effK 7.999/8 with `none`" was measured on an arm WITHOUT Sinkhorn balancing; ours
   run `sinkhorn_iters: 3`, which balances load regardless of `routing_norm`. **That documented
   number does not transfer to any Sinkhorn arm** -- do not cite it for these configs.
2. **The penalty is 0.0185 nats, not "clear".** What `zscore` actually does is SHARPEN PER-TOKEN
   routing -- entropy 0.469 vs 0.589, a 25% relative reduction -- and that is what the 0.019 nats
   buys.

**THE FINDING THAT MATTERS: ROUTING SHARPNESS HAS ALMOST NO LEVERAGE AT THIS SCALE.** A 25% change
in per-token routing entropy costs 0.019 nats; tau across a 5.7x range costs 0.010 nats
(18.6). `routing_norm` is thus ~2x more consequential than tau and both are negligible. This is
consistent with, and independent evidence for, the 400M result that the energy router loses to a
learned gate (18.1) and the 134M result that DENSE matches or beats every MoE (18.3): if the
router's selectivity is worth ~0.02 nats, then most of what the mixture adds is not coming from
routing.

**KEEP `routing_norm: zscore`.** It is better, monotonically, on every checkpoint. Do not switch.

Avg11 at the 8B anchor is being evaluated (`ev_rnorm_none_tok8B`). Per 18.6, DO NOT read an Avg11
gap below ~1.5pp from this protocol as real unless the loss moves with it -- here the loss moves
by 0.019 nats, so any Avg11 difference larger than ~0.5pp should be treated as noise.

## 18.8 ⚖ CALIBRATION: Avg11 IS WORTH ~4.5pp PER NAT — so most 8B Avg11 gaps are NOISE

The tau sweep (18.6) and the `routing_norm` ablation (18.7) both produced Avg11 spreads of
~1.2-1.4pp on loss differences of ~0.01-0.02 nats. To judge whether that is possible, calibrate
against the control's OWN trajectory, where both quantities are known:

```
cmix_134M_hybrid_32B_sparse:  lm_loss 2.9263 @ 8B  ->  2.6544 @ 32B   (dloss = 0.2719)
                              Avg11    43.59 @ 8B  ->    44.82 @ 32B   (dAvg11 = 1.23pp)
              =>  REFERENCE SLOPE ~ 4.5 pp of Avg11 per nat of lm_loss
```

| experiment | dloss | Avg11 PREDICTED at 4.5pp/nat | Avg11 OBSERVED | ratio |
|---|---|---|---|---|
| tau 0.35 -> 2.0 (5.7x) | 0.0096 | **+0.04pp** | 1.39pp | **32x** |
| `routing_norm` none vs zscore | 0.0185 | **+0.08pp** | 1.21pp | **14x** |

**BOTH Avg11 SPREADS ARE NOISE BY AN ORDER OF MAGNITUDE.** Meanwhile loss and perplexity ARE
mutually consistent: `ln(58.25/56.84) = 0.0245` against a 0.0185-nat loss delta, same ballpark.

**RULE FOR THE 8B/32,000-STEP SCREEN: an Avg11 gap is credible only if it is within ~2x of
`4.5 x dloss`. Otherwise report the LOSS and the PERPLEXITY and say Avg11 is unresolved.** The
0.32pp matched-seed floor (measured at 400M on 32B arms) is far too optimistic for this protocol;
the effective Avg11 noise here is of order 1pp or more.

This does NOT weaken the headline results -- it strengthens them by contrast:
* 400M `6G6S` vs `6G6E` is **1.91pp** and comes with a loss/ppl gap of the right size
  (ppl 28.02 vs 30.14, a 7.3% relative difference) -- that is a real effect.
* The 134M dense-vs-MoE orderings were ALREADY flagged as inside the seed floor (18.3), and this
  calibration says the caution was right.
* The prefactor ladder (17.10) was judged on LOSS at matched steps, not Avg11, so it is unaffected.

**Verdicts that survive:** keep `routing_norm: zscore` (better by 0.0185 nats and 1.41 ppl,
monotone on every checkpoint); do not implement learnable temperature (0.0096 nats over 5.7x).
Routing sharpness has almost no leverage at this scale, which is independent evidence for the 400M
and 134M conclusions.

---

### §18.9 (2026-09-22) Cross-table budget confound in the MAIN text, and a ref I broke myself

Landing the generated `tab:main` (32.0B, cmix 70/30) alongside three main-text tables built on the
**7.86B, 100%-web** `iclr_*` grid created a comparison trap: `tab:pure` shows the energy arm at
`Avg11 44.58`, `tab:main` shows the pure arm at `42.00`, and neither caption said the budget or the
corpus. Verified from `configs/iclr_sink/iclr_hop_K32_top2_sink.yml` (not from memory):
`num_training_steps: 30000`, mbs 4, ga 4, and a `datasets:` block with **only** the two
`web-nemotron-cc-hq-p2_*` sets -- no megamath, no finemath.

Fixed in `sec/experiments.tex`: `tab:pure` and `tab:threeway` captions now state 7.86B / 100%-web in
bold and say they are not comparable to `tab:main`. `tab:cost` needed nothing (MACs/token only).
`tab:cmix134m` needed nothing -- it matches `tab:main` to the digit on all four shared rows.

**A dangling ref of my own making.** Moving the old `tab:frontier` to `sec/outtakes.tex` left four
live `\ref{tab:frontier-outtake}`. All resolve, so LaTeX stays silent, but two sat in the MAIN text
resting claims on a table slated for deletion. `experiments.tex:164` was simply mis-pointed -- all
three numbers in its sentence are `tab:pure` rows -- and is repointed. **Three refs still genuinely
need that table** (`experiments.tex:222` aux-loss `43.83->43.12`; `appendix.tex:1602` MMLU range;
`appendix.tex:1990` routing sign), and `outtakes.tex` itself concedes it "remains usable as a K/k
sparsity sweep at 7.86B". So it should be PROMOTED out of outtakes into `appendix.tex` keeping its
label -- left undone because it reverses a placement the user asked for.

**`tab:pure`'s bolded 44.83 is a table maximum, not a measured win.** Final `train-lm_loss`:
energy+Switch 3.1589 (Avg11 44.83), energy+Boltzmann 3.1679 (44.58), GPT+Switch 3.1779 (44.43),
GPT+Switch-3x **3.1387** (44.20). The **best-loss arm scores worst**. The three iso-param arms do
co-order with loss, but across 0.019 nats -- 0.086pp expected at ~4.5pp/nat vs 0.40pp observed. The
prose already says parity and "the same size within noise", so it needed no change.

**Tooling trap, cost me a wrong number.** `compute_avg11.py <run_dir>` globs RECURSIVELY and picked
`ablate/C_proxysel/harness_results_*.json` for `iclr_hop_K32_top2_sink`, returning **44.55**, which I
briefly read as a drift from the published 44.58. Pointing it at `<run_dir>/unsharded` gives
**44.58** and confirms the paper. Pass the EVAL dir, not the run dir, for any arm with an `ablate/`
subtree.

---

### §18.10 (2026-09-22) THE Avg11 GAPS ARE MUCH LARGER THAN THE LOSS SUPPORTS -- read this before quoting any 134M ranking

Final `train-lm_loss` (nats, lower better) against Avg11 (pp), all at 32.0B on the cmix mix:

| 134M arm | Avg11 | lm_loss |   | 400M arm | Avg11 | lm_loss |
|---|---|---|---|---|---|---|
| dense iso-total `6G` | 45.55 | **2.6302** | | Switch deep `6G6S` | 48.83 | **2.4042** |
| Switch FLOP-matched `6G1x6S` | 45.43 | 2.6329 | | Switch FLOP-matched `6G1x6S` | 48.24 | 2.4095 |
| hybrid `6G1x6E` | 44.82 | 2.6544 | | hybrid `6G1x6E` | 47.29 | 2.4748 |
| sandwich `5G1x6E1G` | 43.45 | 2.6597 | | deep energy `6G6E` | 46.92 | 2.4746 |
| dense iso-active `6G` | 45.01 | 2.6640 | | | | |
| pure `1x12E` | 42.00 | 2.9916 | | 1B stacked `8G4E` | 47.79 | 2.4536 |

**At 134M the five non-pure arms lie inside 0.0338 nats.** At the project's ~4.52pp-per-nat slope
(§18.8) that predicts **0.15pp** of Avg11 spread against **2.10pp** observed -- **13.7x**. Applying
the project's own credibility rule (credible only within ~2x of `4.52 * dloss`), every 134M pairing
against the hybrid FAILS:

| vs hybrid | dloss | predicted | observed | verdict |
|---|---|---|---|---|
| dense iso-total | +0.0242 | +0.11pp | +0.73pp | 6.7x -- NOT credible |
| Switch FLOP-matched | +0.0215 | +0.10pp | +0.61pp | 6.3x -- NOT credible |
| dense iso-active | -0.0096 | -0.04pp | +0.19pp | **SIGN INVERSION** (worse loss, better Avg11) |
| sandwich | -0.0053 | -0.02pp | -1.37pp | 57x -- NOT credible |
| pure | -0.3372 | -1.52pp | -2.82pp | **CREDIBLE** |

So the 134M ordering exceeds the 0.32pp matched-seed spread and is presumably reproducible, but it
is **not a language-modelling difference**. The one architectural conclusion the 134M tier supports
is that **pure is genuinely worse** -- by at least 0.32 nats against every other arm (0.36 against
the best). Its WikiPPL 66.08 vs ~40 is consistent with that, so pure's deficit is NOT an eval
artifact; this also finally answers the earlier question about whether pure's gap is real.

**At 400M the loss DOES separate**, and this is where the paper's claim lives: Switch beats the
recurrent energy hybrid by **0.0653 nats** (0.95pp) and the deep energy arm by **0.0704 nats**
(1.91pp), at FLOPwt matched to 0.7% and total params to 0.07% (exact: recurrent pair 0.527% / 0.0658%,
deep pair 0.622% / 0.0014%, from `audit_config`). **The energy deficit in loss TRIPLES with scale,
0.0215 nats at 134M -> 0.0653 at 400M.** That is a far more defensible statement of "the gap widens
with scale" than the Avg11 difference, and it should be the form we quote.

**RETRACTED: "recurrence helps the energy formulation and hurts the learned gate."** In loss,
recurrence moves the energy arm by **+0.0002 nats** (2.4746 -> 2.4748) and the gate by +0.0053 -- it
improves NEITHER. The energy arm's +0.37pp Avg11 buys no measurable modelling gain while costing 26%
more compute, so the reading of it as "repeated descent on an energy" is withdrawn from the caption.

**`tab:main` now carries an `lm_loss` column** so none of this is invisible, and the caption reports
134M as a tie in language modelling with pure as the single real deficit. Caveat to state if
challenged: the 4.52pp/nat slope is measured ALONG ONE RUN'S TRAJECTORY, so using it across
architectures assumes they share a loss-to-Avg11 curve. The sign inversion and the 13.7x factor do
not depend on the slope's exact value.

---

### §18.11 (2026-09-22) Checkpoint saves on the 400M arms went from ~20 s to 15-21 min. ETAs must be computed from OBSERVED CADENCE, not step time.

Around 12:15Z both live 400M arms slowed hard, while both 134M arms were untouched:

| arm | steps/wall-min (12:16-12:33Z) | s/step from the log | save share of wall time |
|---|---|---|---|
| `abl_R` 134M | 235.3 | 0.15 | negligible |
| `abl_C` 134M | 117.6 | ~0.51 | negligible |
| `cmix_400M_sandwich_sparse` | 11.6 | 2.19 | ~80% |
| `abl_H_400M_6G6G_deep_isoactive` | 11.6 | 0.61 | **~88%** |

Evidence it is the SAVE, not the compute: `6G6G` logged step 54,200 at 12:17:52 and step 54,210 at
**12:33:07** -- a 15-minute gap -- and `global_step54200` (2.7G) has mtime 12:32:57. Same arm earlier
the same morning did 200 steps plus a save in ~2.3 min, so saves were ~20 s. `sandwich` shows the
same thing: `global_step60800` 12:11:34 -> `global_step61000` 12:32:21, a 20.8-min gap for 200 steps
of a 2-min workload.

**Not self-inflicted and not a space problem.** Free space actually ROSE across the window
(2603G -> 2807G, so pruning is working), inodes are 83% used with 1.5B free, no write errors in any
stderr, one OMP banner per job (no requeues), and nothing of ours is copying: the watchdog's last
action was 07:04Z and no milestone/rsync/tar process exists. Reads as external GPFS contention. The
134M arms being unaffected fits -- their checkpoints are far smaller than 2.7-4.5G.

**CONSEQUENCE FOR EVERY ETA IN THIS FILE.** An ETA from `remaining_steps x s/step` is now wrong by
up to 8x on a 400M arm, because s/step excludes the save. Earlier today that produced a 13:21Z
estimate for `6G6G` against a real ~22:22Z. Compute ETAs from checkpoint advance over a wall-clock
window instead:

```bash
# steps between two latest_checkpointed_iteration readings, divided by the minutes elapsed
```

Corrected ETAs at 12:33Z: sandwich 12:37Z, `abl_R` 13:25Z, `abl_C` 17:23Z, `6G6G` **22:22Z**. The
Sep 24 00:00Z deadline was 35.4 h away, so all four land with >13 h of margin and NO intervention is
warranted.

**If it degrades further**, the lever is `save_interval` (200 -> 1000 would cut ~5x of the save cost,
taking `6G6G`'s remaining ~9.8 h to ~2 h), but it needs a config edit plus a restart, which costs the
steps since the last checkpoint and risks a bad relaunch. Not worth it at >13 h of margin.

**Monitor thresholds were adjusted**: a staleness alarm at 14 min would now fire constantly on a
healthy 400M arm mid-save. The live watch uses 32 min, above any save observed.

---

### §18.12 (2026-09-22) The unshard lock was node-local, so it never serialised the two eval jobs

`scripts/unshard_once.sh` guards concurrent unsharding with `flock` on a lock file. The lock was in
**`/tmp`, which is node-local**, so it serialised nothing whenever LSF put the two eval jobs
(`ev_*` likelihood and `evg_*` gsm8k_cot, submitted together by `auto_eval_on_finish.sh`) on
different hosts -- the normal case.

Caught live on `cmix_400M_sandwich_sparse`: `ev_*` ran on `p3-r15-n3`, `evg_*` on `p2-r07-n2`, and
both logged `loading checkpoint` at 12:36:01 and 12:36:02, i.e. both passed the
`if [ -f "$OUT/model.safetensors" ]` guard before either had created the file, and both ran
`python -m lm_engine.unshard` into the same directory.

**That particular file is fine, verified rather than assumed:**
- size 800,668,802 B equals `8 + header(2,800) + max_data_offset(800,665,994)` exactly
- 26 tensors, **zero gaps** in the offset map
- no NaN, no Inf, no all-zero tensor; weight stds 0.064-0.252
- (the lone flag, `transformer.h.1.ffwd.moe.sinkhorn_mu_count`, is a scalar counter whose std is
  undefined by construction -- not corruption)

It survived because unsharding is **deterministic**: both processes wrote identical bytes to
identical offsets, so the interleaving was harmless. It is still a race -- a truncation by one
process after the other had advanced would leave a hole, and the size check is the only thing that
would reveal it.

**Fixed:** the lock now lives next to the output on shared GPFS
(`<dir>/.unshard_<basename>.lock`) where every node sees it. Verified idempotent afterwards: a
second call on the already-unsharded sandwich returns `already present` in 14 ms and leaves the
model byte-identical. Backup of the old script at `scripts/unshard_once.sh.bak-20260922`.

**Check this if any eval number ever looks impossible** -- a half-written unshard would produce
plausible-looking garbage, and the integrity check above is four cheap commands.

---

### §18.13 (2026-09-22) `scripts/refresh_paper_tables.sh` — run this instead of regenerating tables by hand

Four separate breakages came from hand-splicing generated tables, so the process is now one command:

```bash
bash experiments/boltzmann-moe/scripts/refresh_paper_tables.sh   # regenerates BOTH, verifies, prints the diff; does NOT commit or push
```

It regenerates `tab:main` and `tab:status`, reports arms complete-but-omitted, diffs caption numbers
against the table body, runs a whole-paper integrity pass, and prints the diff.

**THE TWO GENERATORS EMIT DIFFERENT SHAPES, and this is the trap that broke Table 6 originally.**

| generator | emits | who owns the `\resizebox` |
|---|---|---|
| `gen_table1.py` | marker, `% omitted`, `\resizebox{..}{%`, tabular, `\end{tabular}%`, `}` | the GENERATOR |
| `gen_status_table.py` | marker, tabular, `\end{tabular}` | the **FILE** — the `\resizebox` line sits BETWEEN the marker and the tabular |

So a splice that runs "from the marker" deletes `gen_status_table`'s `\resizebox` and leaves an
unmatched `}`. `splice_generated_table.py` now detects the shape from the generator's last line and
narrows the region to the tabular alone for shape B, leaving marker and resizebox untouched.

**Three guards, each earned by an actual failure, and each negative-tested:**
1. **caption/label guard.** The FIRST version of the splicer assumed the block ran to the line before
   `\end{table}`. In `sec/experiments.tex` the `\caption` and `\label` sit AFTER the tabular, so the
   splice DELETED BOTH — and braces still balanced and environment counts still matched, so every
   check passed. Only a diff caught it. It now refuses if the region contains `\caption` or `\label`.
2. **unescaped-underscore guard.** An arm label with a raw `_` is a LaTeX error in a tabular cell and
   `pdflatex` cannot run here, so it surfaces only on Overleaf. Caught on the abl_R label
   `routing_norm=none`. Verified the guard fires: it exits 1 and leaves the file byte-identical.
   (A whole-document underscore scan is NOT reliable — inline math spanning two lines gives false
   positives. Inside the generated block there is no math, so the scan is exact there.)
3. **braces / environments / shape** as before.

`scripts/check_caption_numbers.py <tex> <label>` diffs numbers asserted in a caption against the
generated body. Derived quantities (gaps, ratios, setup figures) legitimately appear as "missing";
a raw TABLE VALUE in that list means the caption has gone stale. It is what found the FLOPwt
rounding mismatch, both overstated iso-match tolerances, and the false "top four fall inside the
seed spread" claim.

---

### §18.14 (2026-09-22) `abl_C` (`1G1x6E1G`) CANNOT be made iso-active with the hybrid. Read it as iso-total only.

The open question was why the true-sandwich arm carries 98M active against the hybrid's 123M, and
whether raising `I_e` / `n_experts` / `top_k` would fix it. Measured with `audit_config`:

| arm | structure | total | active | FLOPwt |
|---|---|---|---|---|
| `cmix_134M_hybrid_32B_sparse` | `6G1x6E` | 134.25M | 123.24M | 141.72M |
| `cmix_134M_sandwich_32B_sparse` | `5G1x6E1G` | 134.25M | **123.24M** | **141.72M** |
| `abl_C_134M_1G1x6E1G_isototal` | `1G1x6E1G` | 134.25M | 98.46M | 134.64M |

**First, a correction worth having straight: the sandwich row IN `tab:main` is already iso-total,
iso-active AND iso-FLOPwt with the hybrid**, to the digit on all three. It is a clean
placement-only test, and its `-1.37`pp is not confounded by parameter count. The 98M figure belongs
to `abl_C`, a different arm.

**Second, `top_k` does not fix `abl_C`.** Sweeping it at fixed `I_e = 53248`:

| `top_k` | total | active | FLOPwt |
|---|---|---|---|
| 2 | 134.25M | 98.46M | 134.64M |
| 3 | 134.25M | 101.02M | 149.97M |
| 4 | 134.25M | 103.58M | 165.31M |
| 6 | 134.25M | 108.69M | 195.98M |

At `top_k=6` active is still **15M short** of 123.24M while FLOPwt is **38% over** target. The
reason is that the shortfall is not in the mixture at all: `1G1x6E1G` has **two** dense GPT blocks
where the hybrid has six, and a dense block's parameters are **fully** active whereas expert
parameters are only `k/K` active. Four missing dense blocks are ~25M of active parameters that no
amount of expert widening recovers. `I_e` is already widened 3.25x (53,248 vs 16,384) just to reach
iso-total.

**So iso-total and iso-active are mutually exclusive for this structure**, and `abl_C` must be
reported as what it is: an iso-**total** arm that trades dense capacity for expert capacity. Its
result answers "what happens if four dense GPT blocks are replaced by a wider recurrent energy block
at constant total parameters", NOT "does block placement matter" -- that question is already
answered cleanly by the `5G1x6E1G` row. Do not read `abl_C` against the hybrid as a placement test,
and say in the caption which quantity is held fixed.

---

### §18.15 (2026-09-22) 400M sandwich result: Avg11 44.52 — and it is NOT a controlled comparison

`cmix_400M_sandwich_sparse` finished the full 32B (`tok32B_step61035_actual32.00B`) and evaluates to
**Avg11 44.52 / MMLU 26.18 / GSM8K-CoT 2.12 / WikiText 41.13 word-ppl**, final `lm_loss` **2.6955**.
Against the 400M hybrid that is **-2.77pp and +0.221 nats**, and its perplexity is worse than every
134M arm.

**Do not read it as a placement effect.** The two arms called "sandwich" are different structures:

| | structure | total | active | FLOPwt | lm_loss | Avg11 |
|---|---|---|---|---|---|---|
| 134M hybrid | `6G1x6E` | 134.25M | 123.24M | 141.72M | 2.6544 | 44.82 |
| **134M sandwich** | `5G1x6E1G` | 134.25M | **123.24M** | **141.72M** | 2.6597 | 43.45 |
| 400M hybrid | `6G1x6E` | 399.78M | 219.43M | 299.38M | 2.4748 | 47.29 |
| **400M sandwich** | `1G1x4E1G` | 400.33M | **156.54M** | **217.20M** | 2.6955 | 44.52 |

The **134M** sandwich is matched on total, active AND FLOPwt, so its `-1.37`pp genuinely isolates
block placement. The **400M** sandwich is iso-**total** only: **29% fewer active parameters and 27%
less compute per token**. Its deficit is mostly a smaller-effective-model effect.

This is the same structural impossibility as `abl_C` (§18.14): a `1G...1G` arm has two dense blocks
where the hybrid has six, and dense parameters are **fully** active while expert parameters are only
`k/K` active, so the missing dense capacity cannot be bought back by widening experts or raising
`top_k`. Confirmed numerically on the 134M analogue: `top_k` 2->6 moves active 98.46M -> 108.69M
(still 15M short of 123.24M) while FLOPwt goes **38% over** target.

**In the paper:** the row is in `tab:main` with its FLOPwt (217.2M) and `lm_loss` (2.6955) visible, so
it is self-documenting, and the caption now (a) drops the claim that rows within a group "differ only
in architecture" -- they share budget and data but not always compute -- and (b) states explicitly
which of the two sandwiches is the controlled one. **Any future `1G..1G` arm needs the same warning.**

---

### §18.16 (2026-09-22) `routing_norm: none` at FULL 32B — keep `zscore`, and the Avg11 sign flipped between budgets

`abl_R_134M_hyb_rnorm_none` completed the full 32B (`tok32B_step122070_actual32.00B`). It is a clean
one-variable ablation -- `diff` against `cmix_134M_hybrid_32B_sparse` shows **only**
`routing_norm: zscore -> none`, plus `save_path` and the wandb name.

| `routing_norm` | Avg11 (pp) | MMLU (%) | WikiText (word-ppl, lower better) | final `lm_loss` (nats, lower better) |
|---|---|---|---|---|
| `zscore` (shipped) | 44.82 | **25.57** | **41.06** | **2.6544** |
| `none` | **44.92** | 24.74 | 42.36 | 2.6747 |

`none` wins on ONE of four measures, by **0.10pp** of Avg11 -- inside the 0.32pp matched-seed spread
-- while losing **0.0203 nats** of loss, **1.30** of perplexity and **0.83** of MMLU. **Verdict
unchanged: keep `routing_norm: zscore`.** But the justification is loss and perplexity, not Avg11.

**This is the cleanest demonstration yet of why §18.10's loss column matters.** The Avg11 verdict
INVERTED between budgets while the loss verdict replicated almost exactly:

| budget | Avg11 delta (none - zscore) | loss delta (none - zscore) |
|---|---|---|
| 8B | **-1.21pp** (none looked much worse) | +0.0185 nats |
| 32B | **+0.10pp** (none looks better) | +0.0203 nats |

Same sign and nearly the same magnitude in nats; opposite signs in Avg11. Anyone quoting the 8B
Avg11 figure as evidence would have had the right conclusion for the wrong reason, and anyone
quoting the 32B Avg11 figure would reach the wrong conclusion.

**Routing health does not distinguish them, and CLAUDE.md's concern does not bite here.**
`load_effective_n_experts` is **15.67 of 16** with `zscore` and **15.86 of 16** with `none`, so
neither collapses and `none` is if anything slightly better balanced. Note this metric is the
BATCH-LOAD measure, not the per-token entropy that CLAUDE.md's routing-collapse table describes --
a high value means loads are even across experts, which is the desirable outcome. The balancing is
being done by **Sinkhorn**, not by the logit normalisation, which is why removing the normalisation
does not open the routing up.

---

### §18.17 (2026-09-22) Independent corroboration: the pure-recurrent penalty is ~0.32 nats in BOTH waves

`sec/appendix.tex` §`app:findings` item 4, written from the **7.86B / 100%-web** wave, attributes the
pure-recurrent deficit to the architecture and puts it at **~0.32 nats** (against 0.025-0.059 nats
for the selection rule, a ~7.5x ratio).

My 32B measurement (§18.10) was made independently, on the **32.0B / 70-30 web-math** mix, and gives
**0.328-0.361 nats** for `pure 1x12E` against every other 134M arm (0.3276 vs dense iso-active,
0.3614 vs dense iso-total). Different wave, different corpus, different arms -- **same magnitude**.

That is the strongest single result in the project: the pure-recurrent penalty is large, reproducible
across corpora, and it is the one architectural conclusion that passes the credibility rule at both
budgets. Everything else in the 134M tier sits inside 0.034 nats and does not.

**Also fixed in this pass:** `app:findings` opened with "a single exploratory wave" but never named
its budget or corpus, so its 44.58 could be read against `tab:main`'s 42.00. It now states 7.86B /
100%-web in bold, disclaims comparison to `tab:main`, and says the findings are internally
controlled rather than the paper's headline numbers. That completes the budget-labelling sweep begun
in §18.9: `tab:pure`, `tab:threeway`, `app:findings` all labelled; `tab:cost` needs none (MACs/token
only); `tab:cmix134m` and `tab:status` are already 32B.

---

### §18.18 (2026-09-22) The 400M deep trio is iso-FLOP to 0.62%, and the DENSE arm beats the energy MoE while storing 40% fewer parameters

`abl_H_400M_6G6G_deep_isoactive` completed 32B (`tok32B_step61035_actual32.00B`), final `lm_loss`
**2.4541**. That closes a three-way comparison that is genuinely controlled -- `audit_config` puts the
FLOPwt spread at **1.49M = 0.62%**, and the energy and dense arms are iso-ACTIVE to the byte:

| arm | blocks | total | active | FLOPwt | `lm_loss` (nats, lower better) |
|---|---|---|---|---|---|
| `abl_H_400M_6G6S_deep` | `6G` + `6S` | 399.86M | 239.50M | 239.50M | **2.4042** |
| `abl_H_400M_6G6G_deep_isoactive` | `12G` dense | **238.02M** | 238.02M | 238.02M | 2.4541 |
| `abl_H_400M_6G6E_deep` | `6G` + `6E` | 399.85M | 238.02M | 238.02M | 2.4746 |

**Switch 2.4042 < dense 2.4541 < energy 2.4746.** Pairwise: Switch beats dense by 0.0499 nats, dense
beats energy by 0.0205, Switch beats energy by 0.0704.

**The strongest form of the negative result.** The dense arm stores **238M total against the MoE
arms' 400M** -- 40% fewer parameters -- at the SAME compute per token and the SAME active count, and
it still beats the energy mixture. So at 400M the 162M of extra stored expert parameters are not
merely unhelpful, they are worse than not having them. This is a cleaner statement than any Avg11
comparison and does not depend on the 4.52pp/nat slope.

**What is consistent across both scales** is that energy trails Switch, and the gap grows:
0.0215 nats at 134M -> 0.0653 (recurrent) and 0.0704 (deep) at 400M. **What is NOT consistent** is
dense vs Switch: at 134M dense iso-total is best (2.6302 vs Switch 2.6329), at 400M Switch is best
(2.4042 vs dense 2.4541). Do not claim a scale trend for dense-vs-Switch from two points that
disagree.

Note both eval jobs landed on different hosts again (`p6-r08-n2`, `p6-r28-n1`), so this is the first
real exercise of the GPFS unshard lock from §18.12; verify the unsharded file's integrity when the
eval lands.

---

### §18.12a (2026-09-22) CORRECTION to §18.12 — the GPFS flock fix did NOT work. The real fix is atomic rename.

§18.12 said moving the `unshard_once.sh` lock from node-local `/tmp` onto shared GPFS fixed the
concurrent-unshard race. **It did not, and the verification was inadequate**: I only checked that a
second call on an already-unsharded directory short-circuits, which exercises the
`-f model.safetensors` test and never touches the lock.

**Measured refutation.** `abl_H_400M_6G6G_deep_isoactive`'s two eval jobs landed on different hosts
(1856852 on `p6-r08-n2`, 1856853 on `p6-r28-n1`) and logged `loading checkpoint` at **13:48:18.449**
and **13:48:19.062** -- 613 ms apart, i.e. concurrently. Both ran `lm_engine.unshard` into the same
directory again. `flock` on GPFS does not give cross-node mutual exclusion here.

**v3 does not rely on locking for correctness at all.** Each caller unshards into its own private
`${OUT}.tmp.<host>.<pid>` and then moves each file into place with `mv -n`, which within one
filesystem is `rename(2)` and therefore atomic, and which never clobbers a file another writer
already placed. `model.safetensors` is moved LAST, so the directory is only advertised as complete
once everything else is in place -- and `-f model.safetensors` is exactly the readiness flag every
caller tests. Two concurrent callers now duplicate ~20 s of work instead of interleaving writes.

**Tested the way v2 should have been:** four concurrent callers against one fresh target, with a stub
writer slowed down to widen the race window. Result was a single complete file of exactly the
expected length and content, and zero leftover temp directories. The `-f` check at the top still
short-circuits the common case where one job starts after the other has finished.

Both previously produced files are fine, verified rather than assumed -- `sandwich` 800,668,802 B /
26 tensors / 0 gaps, `6G6G` 476,039,184 B / 74 tensors / 0 gaps, sizes matching their safetensors
headers exactly. They survived because unsharding is deterministic, so the two processes wrote
identical bytes at identical offsets. That is luck, not a guarantee.

The useless v2 lock file was deleted. Backup of v1 remains at `scripts/unshard_once.sh.bak-20260922`.

---

### §18.19 (2026-09-22) CORRECTION to §18.10 — the "13.7x" figure used the WRONG SLOPE. Fit the cross-arm slope instead.

§18.10 concluded the 134M Avg11 spread was **13.7x** what the loss accounts for, and therefore "not a
language-modelling difference". **That arithmetic used the within-RUN slope of 4.52pp/nat** (measured
along the control's own training trajectory, §18.8), which is the wrong instrument for comparing
different architectures at a fixed budget.

**Regress Avg11 on final `lm_loss` across the arms of a tier** (all 32.0B, same mixture):

| tier | slope | R^2 | residual s.d. |
|---|---|---|---|
| 400M, 6 arms | **-13.50** pp/nat | **0.951** | 0.30pp |
| 134M, 6 arms | **-8.58** pp/nat | 0.761 | 0.62pp |

So Avg11 **does** track loss, about **2-3x more steeply across arms than along one run**. The
"13.7x" is an artifact of using 4.52 where 8.58 belonged, and the claim that the 134M ordering is
"not a language-modelling difference" was too strong.

**With the right slope the picture is sharper, not vaguer.** Residuals from the tier fits:

| tier | arm | residual |
|---|---|---|
| 134M | **sandwich `5G1x6E1G`** | **-1.32pp** |
| 134M | hybrid / pure / dense-isoactive / Switch / dense-isototal | +0.01 to +0.53pp |
| 400M | all six arms | -0.43 to +0.53pp |

1. **400M is solid.** Every arm within 0.53pp of the line -- loss and Avg11 agree, so the ordering
   (Switch < dense < energy in loss) is trustworthy and is not a metric artifact.
2. **134M is unresolved, not refuted.** Five arms sit inside 0.0338 nats; refitting on those five
   alone gives R^2 = 0.45 and an unstable slope (-36pp/nat), i.e. the loss range is too narrow to
   order them. Claim no ranking there -- but do not claim their ordering contradicts loss either.
3. **`pure` sits ON the line** (+0.08pp) and is >=0.32 nats behind everything: its deficit is real
   and is a genuine language-modelling deficit, corroborated at ~0.32 nats by the independent
   7.86B wave (§18.17).
4. **NEW, and the most specific finding of the tier: the sandwich is the one arm that misses the
   line**, by **-1.32pp** (6-arm fit) / -0.99pp (5-arm fit), 3-4x the 0.32pp matched-seed spread,
   for a loss penalty of only 0.0053 nats. Moving the energy block one position earlier costs
   downstream accuracy **over and above** anything visible in the loss. That is a placement effect
   on task transfer, not on language modelling.

**Method rule going forward: never convert nats to Avg11 with the within-run slope.** Fit the slope
on the arms being compared, report R^2 and the residual s.d., and judge an arm by its RESIDUAL
against the fit rather than by a raw gap. `tab:main`'s caption now does this.

---

### §18.20 (2026-09-22) `tab:main` is now INDEPENDENTLY verified, cell by cell — `scripts/audit_table1.py`

Regenerating a table proves nothing about correctness: `gen_table1.py` and the table share code, so a
bug in the generator writes the same wrong number both times. `scripts/audit_table1.py` reads the
**.tex that will actually be submitted** and recomputes every cell by a DIFFERENT path:

| column | independent source |
|---|---|
| Active/Total, FLOPwt | `energy_ff_paramcount.audit_config` on the arm's config |
| **Avg11** | the eleven task scores re-averaged **in the audit itself**, from the step-pinned eval JSON -- deliberately NOT via `compute_avg11.py`, so a bug there cannot hide |
| `lm_loss` | last `train-lm_loss` in the arm's own training stderr (excluding `ev_`/`evg_` logs) |
| WikiPPL / MMLU / GSM8K | read straight out of the same eval JSON |

**Result: 13 rows re-derived, 0 mismatches**, across all seven checked columns. Tolerances are tight
(6e-5 on `lm_loss`, 6e-3 on accuracies and perplexity, 0.06M on FLOPwt, exact string on Active/Total).

It is wired into `refresh_paper_tables.sh` as step 4b and **exits non-zero on any mismatch**, printing
`DO NOT PUSH`. It also fails loudly on a row it cannot map, so adding an arm to `gen_table1.GROUPS`
without adding it to `audit_table1.MAP` is caught rather than silently skipped.

Run `bash scripts/refresh_paper_tables.sh` after any new arm completes; it regenerates both tables,
audits every cell, checks the whole paper, and prints the diff. It never commits and never pushes.

---

### §18.21 (2026-09-22) `tab:status` spot-audited: 21 rows with results, 0 real mismatches. Both flags were the auditor's own bug.

Ran the same independent re-derivation against `tab:status` (24 rows, 21 reporting Avg11). Two flags,
**both false positives of the audit script, not defects in the table**:

1. The header row `Arm & arch & ...` was parsed as data. Cosmetic.
2. `134M w1w2, dense, surrogate router` -- table says **44.16**, the audit computed 45.01.
   **The table is right.** Two arms share the base label `'134M w1w2'` in
   `gen_status_table.py` and are distinguished only by the routing suffix the generator appends:

   | config | printed as | Avg11 |
   |---|---|---|
   | `cmix_134M_hyb_w1w2_surrMLP_32B` | `134M w1w2, dense, surrogate router` | **44.16** |
   | `cmix_134M_hyb_w1w2_sparse_surr_32B` | `134M w1w2, sparse(surrogate)` | **45.01** |

   The audit matched by longest label prefix, which cannot separate them, so it compared the first
   row against the second arm's score. 44.16 also matches the value recorded in CLAUDE.md for that
   arm, independently.

**Lesson, and it is why `audit_table1.py` was built with an explicit `MAP`:** auditing by label prefix
is unsafe whenever two arms share a label stem. Anything that audits `tab:status` properly needs the
same explicit config-per-row mapping. Not built -- `tab:status` is an appendix status table and the
spot check found nothing wrong; `tab:main`, which carries the paper's claims, is fully audited
(§18.20).

**One substantive datapoint surfaced by this:** `cmix_134M_hyb_w1w2_sparse_surr_32B` scores **45.01**,
i.e. **above** the hopfield hybrid's 44.82, at 134M/32B. That is inside the 0.32pp seed spread so it
is not a win, but it does mean the w1w2 expert form with a sparse surrogate router is NOT behind
hopfield at this scale -- worth remembering before any claim that hopfield is the better expert form.
Note also §15.1's warning that expert form and selector are confounded across the grid.

---

### §18.22 (2026-09-22) VALIDATED: an 8B read on LOSS gives the right ablation verdict. ~4x more ablations per GPU-day.

Ranked every 32B arm by median `train-lm_loss` in a 1000-step window around its 8B crossing, and
compared that ranking to the final 32B ranking:

| tier | arms | exact rank matches | Spearman rho |
|---|---|---|---|
| 134M (8B = step 30,517 at 262,144 tok/step) | 7 | **7/7** | **1.000** |
| 400M (8B = step 15,259 at 524,288 tok/step) | 6 | 4/6 | **0.943** |

The 400M inversion is only the two Switch arms (`abl_B_400M_6G1x6S` 2.4095 and `abl_H_400M_6G6S_deep`
2.4042), which differ by **0.0053 nats** at 32B -- effectively tied. **Rule: an 8B loss read is
authoritative except between arms closer than ~0.005 nats.**

**Avg11 at 8B is NOT usable for this** and would have sent us the wrong way: the `routing_norm`
delta was **-1.21pp at 8B and +0.10pp at 32B** (sign flip) while the loss delta reproduced within
10% (+0.0185 -> +0.0203 nats). §18.16.

**It needs NO checkpoint -- it reads the training log.** That matters because milestone capture is
unreliable: `abl_C` and `abl_D` both LACK their `tok8B` anchors (abl_C has 16B and 24B only, abl_D
has 32B only) even though `abl_C`'s first job covered steps 10-61,890 and so did cross step 30,517.
The likely cause is the backup script, which can only hard-link a checkpoint that still exists, and
`max_to_keep: 2` prunes an 8B checkpoint long before it runs. **Do not depend on the tok8B
directory; depend on the log.**

Precision: a 1000-step window gives ~100 samples at sd ~0.022 nats, so **SE ~0.0022 nats** -- an
order of magnitude below the effects we act on (0.02-0.07 nats).

**Tool: `scripts/rank_arms_at_milestone.py`**
```bash
python scripts/rank_arms_at_milestone.py --tokens 8 --tok-per-step 262144 ARM [ARM ...]
```
It prints the milestone ranking, each arm's sd/n/SE, its final loss for comparison, and the smallest
adjacent gap, with a reminder to treat sub-0.005-nat pairs as unranked.

**Operational consequence: stop ablations at 8B unless the arm is a paper row.** 8B is a quarter of
32B. Of the 134M ablations run to full budget today, every one would have reached the same verdict at
8B.

---

### §19.1 (2026-09-22, URGENT) Dataset rescue — the owners were to delete `/proj/datasets/granite-4-datasets-megatron-merged` today

**Sizes.** Four datasets our configs use: **3,145 GB** total. But the `.bytes` files (504G + 501G =
**1,005 GB**) are read ONLY by `lm_engine/data/byte_megatron_dataset.py`; every cmix/iclr config uses
`class_name: MegatronDataset`, so the real requirement is **2,143 GB** (`.bin` + `.idx`), of which
**2,000 GB** is the two web `.bin`.

| | .bin | .idx | .bytes |
|---|---|---|---|
| `web-nemotron-cc-hq-p2_0` | 1002G | 7.1G | 504G |
| `web-nemotron-cc-hq-p2_1` | 998G | 6.9G | 501G |
| `megamath-web-pro_0` | 49G | 0.3G | -- |
| `finemath-3plus-rewritten_0` | 79G | 0.4G | -- |

**THE TRAP IN THE SPACE NUMBERS.** `df /u/ndehmamy` reports **79 TB avail** on a 100 TB fileset. That
is fileset capacity shared by all users. `mmlsquota -u $USER ess6000-1` shows fileset **`user.home`:
27 GB used, quota 90, limit 100** -- a hard **100 GB** cap. Home is unusable for bulk data. By
contrast the `dmfexp` and `datasets` filesets have per-user quota **0 = unlimited**, so there the
FILESET capacity binds: dmfexp ~3.5-4.5 TB free (oscillates +-1 TB from colleagues), datasets ~7.8 TB.
`/opt/nvme` has 26 TB but is NODE-LOCAL and worthless for preservation. The underlying ess6000-1 has
**2.3 PB** free -- the 350 T dmfexp fileset cap is the only blocker.

**WHAT WAS DONE**

1. **Hard links, zero space, instant** -- `/proj/datasets/ndehmamy-dataset-rescue/`. All 14 files of
   the four datasets, verified `links=2`, same inode, matching size, and the fileset's avail did NOT
   move. Works because the source is on the same fileset. **This defeats an `rm` by the owners** --
   the inode survives while our link holds it. It does NOT defeat the fileset being removed, nor
   truncation/overwrite in place.
2. **Real copies on the dmfexp fileset** (`/proj/dmfexp/nima/datasets-rescue/`), 142 GB: both math
   `.bin`+`.idx` and all four `.idx`. Verified independently -- 10/10 sizes match, **no shared
   inodes** (so they are genuine copies), md5 match on both math `.idx`.
3. **THE TOKENIZER, which was the real near-miss.** `/proj/datasets/tokenizers/granite-4.0-tiktoken`
   is **14 MB** and referenced **492 times** across configs, and it sits on the same at-risk fileset.
   Without it nothing is reproducible. Now in THREE places -- dmfexp, `~/dataset-rescue/`, and
   hard-linked -- with `tokenizer.json`, `vocab.json` and `merges.txt` md5-verified identical.
4. `reasoning-megatron` (3.5 GB, 30 files) copied to dmfexp.

**STILL A DECISION: the two web `.bin`, 2,000 GB.** `scripts/rescue_datasets_tier2.sh` is staged and
NOT submitted. It refuses to drive dmfexp below an 800 GB floor. **Only worth running if the whole
`datasets` fileset is at risk** -- against an owner `rm`, the hard links already cover it for free.
Fallback if the web data is lost: Nemotron-CC is public, so it is re-tokenisable with the (now
rescued) Granite-4 tokenizer, at the cost of a tokenisation run.

**Bug worth remembering:** the first tier-1 job exited instantly because the file list had three
names per line and `while read -r f` took a whole line as one filename. Nothing was copied and no
space was used. Iterate with `for f in $FILES`.

---

### §19.2 (2026-09-22) A SHARED, 275 GiB self-sufficient copy of the cmix datamix — `/proj/dmfexp/datasets-shared/granite-4-cmix-subset/`

**The insight that made this cheap.** The corpus is 570.7B tokens / 2.1 TB (int32, 4 B/token, read
from the `.idx` headers), but a 32B-token run only *consumes* **119 GiB** — 4.2% of each web shard,
37% of megamath, 23% of finemath. So preserving the whole thing is 18x more than a run needs.

| | tokens | size | provenance |
|---|---|---|---|
| `web-...-p2_0` | 20.0B | 74 GiB | **prefix subset** of 268.7B |
| `web-...-p2_1` | 20.0B | 74 GiB | **prefix subset** of 267.9B |
| `megamath-web-pro_0` | 13.0B | 48 GiB | whole |
| `finemath-3plus-rewritten_0` | 21.1B | 79 GiB | whole |
| tokenizer | -- | 14 MiB | whole, md5-verified |

20B per web shard is **1.79x** the 11.2B a 32B run draws, single-epoch, good to ~57B tokens.

**WHY THE SUBSETS ARE EXACT AND NOT A RE-TOKENISATION.** Sequences sit back-to-back in the `.bin` in
index order, and `_IndexWriter._sequence_pointers` recomputes pointers from lengths starting at 0.
Verified on the real files: `pointers[i] == sum(lengths[:i]) * itemsize` for the first 100k sequences
of both megamath and p2_0. So keeping the first S sequences (rounded UP to a document boundary) means
new `.bin` = a plain BYTE PREFIX, new `.idx` = the truncated arrays. Builder:
`scripts/build_dataset_subset.py`, with `--verify` reloading the result and comparing documents
against the source (first/middle/last all byte-identical).

**WHAT IT IS NOT FOR.** `gpt_dataset.py` builds a `shuffle_index` that is a random permutation over
the whole document set, so a published arm sampled documents scattered across the full 268B-token
shard. A subset changes which documents exist and therefore which are sampled. **A rerun on the
subset does NOT reproduce a published arm** -- use the hard links in
`/proj/datasets/ndehmamy-dataset-rescue/` for that.

**SHARING.** `grp_ebm` is an **LSF user group, not a POSIX group**, so it cannot be used for file
permissions. The usable groups are `proj_dmfexp` (100+ members incl. `bharat`) and `proj_datasets`.
Both filesets are `drwxrws--- root:proj_*`, so **group-readable is the maximum useful** -- world bits
achieve nothing because non-group users cannot enter the fileset. Note `/proj/dmfexp/nima` is
`drwx--S---`, so anything under it is private; that is why the shared copy lives at the FILESET ROOT
(`/proj/dmfexp/datasets-shared`, mode 2775 + setgid) rather than in my tree, which would have exposed
checkpoints and logs. The hard-linked files were deliberately NOT chmod'd -- they share inodes with
the owners' originals, so a chmod on a link changes THEIR file (they are already `-rwxrwxrwx`).

**Usage:** `configs/cmix/cmix_134M_hybrid_32B_sparse_SHARED.yml` is ready to run (12 changed lines:
paths, tokenizer, and a FRESH `data_cache_path` -- the Megatron blend index is keyed on the mix, so
reusing the old cache dir collides or silently rebuilds). A `README.md` sits in the shared directory.

**TWO MEMORY BUGS IN MY OWN BUILDER, worth knowing for any tool that touches these indices.** The
first build died at an 8 GB LSF limit. p2_0 has **379,955,333 sequences**, so
`sequence_lengths.astype(np.int64)` is 3.0 GB, a full `np.cumsum` another 3.0 GB, and `.tolist()` on
the ~28M kept sequences would add ~1 GB of Python int objects. Fixes: accumulate the cumsum in 16M
chunks, keep the mmap int32 view instead of casting, pass numpy arrays to `_IndexWriter` (it tolerates
them -- it uses `len()`, `max()`, `np.array()` and `.item()`), and **slice the sorted document index
rather than masking it** (`D[D <= S]` faults in all 3.0 GB of `document_indices`; use
`D[:searchsorted(D, S, 'right')]`). Peak afterwards: **391 MB**. Re-validated byte-identical after
every change.

---

### §19.3 (2026-09-22) The shared subset TRAINS — smoke test passed end to end

`configs/cmix/_smoke_shared_subset.yml`, 40 steps, 2 GPUs, job 1862284: **DONE in 141 s**, zero
errors. `train-lm_loss` 7.9407 -> 7.7970 -> 7.7019 -> **7.5841** over steps 25-40, checkpoint saved
at 40. All four datasets were read from `/proj/dmfexp/datasets-shared/granite-4-cmix-subset/` and the
tokenizer from the shared copy. tokens/step from the log: `32.4467e9 * 0.1745 / 86400 = 65,542`
= 65,536 as expected for 2 GPU x mbs 4 x ga 2 x 4096.

**The Megatron blend index is now cached** at `/proj/dmfexp/datasets-shared/.cache/` (524 MB, 42
`.npy`), so real runs on the shared data start without paying the index build.

**Two process failures worth remembering:**
1. The first smoke submission died with `torchrun: command not found` because I hand-rolled the
   bsub and skipped the venv. CLAUDE.md says plainly: NEVER hand-roll a training bsub, use
   `scripts/bsub/submit_train.sh <name> <cfg> <gpus>` -- it sources the venv, sets PYTHONPATH and
   handles the node shape / ptile / blaunch rules. Doing that worked first time.
2. The watcher meant to auto-submit the smoke test after the build never fired. It waited for
   `bjobs` to return EMPTY for the build jobs, but **LSF keeps DONE/EXIT rows visible for its
   CLEAN_PERIOD** -- `bjobs 1861791` still printed `DONE` 18 minutes later. Wait on the STAT being
   DONE/EXIT, never on the row disappearing. The build had been finished and verified the whole
   time; only the trigger was broken.

### §19.4 (2026-09-22) The datamix caps out at 86B tokens, and that is NOT a rescue artifact

Asked what a 128B run would need. Storage is the easy part (+188 GiB of web prefixes, taking the
shared copy from 292 GB to ~480 GB). The real constraint:

| source | total tokens | weight | single-epoch ceiling |
|---|---|---|---|
| `megamath-web-pro_0` | **13.0B** | 0.15 | **85.9B** <- BINDING |
| `finemath-3plus-rewritten_0` | 21.1B | 0.15 | 139.4B |
| `web-...-p2_0` | 268.7B | 0.35 | 760.1B |
| `web-...-p2_1` | 267.9B | 0.35 | 757.8B |

**megamath has only 13.0B tokens in existence** -- we already copied all of it. A 128B run needs
19.4B of it (after the 99% train split), i.e. **1.49 epochs**; finemath lands at 0.92, essentially
exhausted. So a 128B run on the 0.35/0.35/0.15/0.15 blend would repeat math data **even with the
original directory fully intact**, and it breaks the setup claim that "no arm revisits a document".

Options: reweight (0.425/0.425/0.075/0.075 lifts the ceiling to 173B but breaks comparability with
every 32B arm); accept and state the 1.49 epochs; or tokenise more math (MegaMath has splits beyond
`web-pro`, FineMath has larger variants -- needs the rescued Granite-4 tokenizer, a real job not a
copy). **Decision pending with the user.**

---

### §19.5 (2026-09-22) EVERY KNOB, IN ONE TABLE — none of them improves Boltzmann MoE

Asked directly whether any knob made the energy MoE significantly better. Measured against the
shipped hybrid (`cmix_134M_hybrid_32B_sparse`, `lm_loss` 2.6544, Avg11 44.82), all at 134M/32B on the
cmix mix. **Negative dloss = better than shipped.**

| what changed | dloss | dAvg11 |
|---|---|---|
| **no energy at all** (`abl_D`, dense GPT) | **-0.0242** | +0.73 |
| **learned gate** instead of Boltzmann (`abl_B`, `6G1x6S`) | **-0.0215** | +0.61 |
| expert form -> w1w2, dense surrogate | -0.0193 | -0.66 |
| expert form -> w1w2, sparse surrogate | -0.0108 | +0.19 |
| no mixture at all (`abl_E`, base EGPT) | -0.0012 | **+1.05** |
| `routing_norm` zscore -> none (`abl_R`) | **+0.0203** | +0.10 |

**Every variant that improved loss did so by REMOVING or REPLACING the Boltzmann mechanism.** The one
knob purely internal to it, `routing_norm`, got WORSE when changed -- the shipped value was already
the better of the two. The best 134M arm on Avg11 is `abl_E` at **45.87**, which has no mixture at all.

Knobs tested at shorter budgets, all inert or harmful:
* **tau** 0.35 and 2.0 -- inert (~0.01 nats), stopped at 32k and 38k of 122k steps.
* **gradient prefactor** -- the mathematically EXACT one costs **0.35 nats**; shipped
  `sqrt_consistent` is best. All three ladder arms stopped at 4k steps once decided.
* **`proj` unconstrained -- inert, -0.0020 nats.**

**A PAIRING TRAP, and I nearly reported it wrongly.** Comparing `abl_I_134M_w1w2_sparse_surr_projUncon`
(2.6416) against the HOPFIELD hybrid (2.6544) credits `proj` with -0.0128 nats -- but that pairing
also changes the expert form AND the router. Against its true single-knob sibling
`cmix_134M_hyb_w1w2_sparse_surr_32B` (2.6436, same w1w2, same sparse surrogate) `proj` is worth
**-0.0020 nats**. Always diff against the arm that differs in ONE thing.

**THE ONE OPEN THREAD.** `w1w2` + sparse surrogate is 0.0108 nats better than hopfield + proxy and is
the only energy variant ahead of the shipped hybrid on BOTH loss and Avg11 (45.01 vs 44.82). It
**cannot be attributed**: per §15.1 every hopfield arm uses the proxy router and every w1w2 arm the
surrogate, so expert form and selector are confounded across the whole grid. **One arm breaks it --
hopfield experts + surrogate router.** That is the only energy-side experiment still worth GPU time.

### §19.6 The 8B rule, stated so it cannot be misread

Two claims that sound contradictory are about DIFFERENT METRICS:

| measured at 8B | predicts the 32B result? |
|---|---|
| `train-lm_loss` | **YES** -- Spearman rho 1.000 at 134M (7/7 arms), 0.943 at 400M |
| `Avg11` | **NO** -- `routing_norm` was -1.21pp at 8B and **+0.10pp** at 32B, a sign flip |

**Rule: compare at 8B, on loss, never on Avg11.**

---

### §19.7 (2026-09-22) Web subsets extended to 50B tokens each — 128B runs are now possible. DONE and validated.

User approved the extension. Both web shards rebuilt as 50B-token prefix subsets, verified, swapped:

| shard | seqs | tokens | size | % of source |
|---|---|---|---|---|
| `web-...-p2_0` | 58,223,086 | 50,000,000,150 | 186.3 GiB | 18.61% |
| `web-...-p2_1` | 58,213,008 | 50,000,000,729 | 186.3 GiB | 18.67% |

Shared tree **522 GB** (from 292 GB, so +230 GB -- I quoted 188 GiB as the bare minimum for 45.3B and
deliberately went to 50B for 1.10x headroom, flagged to the user before starting). dmfexp 3,886 GB free.

**Validated end to end:** the builder verified each new file against the source before swapping
(lengths a true prefix, idx/bin agree, 6/6 sampled docs byte-identical); an independent pass then
re-read all four `.idx` (50.000B / 50.000B / 13.011B / 21.119B, all idx/bin agreeing, zero `.new.*`
leftovers); and a 40-step smoke test on 2 GPUs trained cleanly -- `lm_loss` 7.8557 -> **7.6329**, 0
errors -- rebuilding the blend cache (959 MB, 42 `.npy`).

**Two design choices that paid off:**
1. **Build to `*.new.*`, verify, then swap.** The working 20B data stayed valid throughout, so a
   preemption or a verification failure would have cost nothing.
2. **Delete the blend cache on success.** It indexed the OLD 20B document set; leaving it would
   either error or silently sample the wrong documents. Same hazard class as reusing a
   `data_cache_path` across datamixes.

**Capability now, and the ceiling is NOT ours:**

| budget | web | megamath | finemath | verdict |
|---|---|---|---|---|
| 32B | 4.4x | 2.7x | 4.4x | comfortable |
| **86B** | 1.6x | **1.00x** | 1.6x | **single-epoch ceiling** |
| 128B | 1.10x | **1.49 epochs** | 0.92x | user accepted the repeat |

`megamath-web-pro_0` holds only **13.0B tokens in existence** and all of it is copied, so at weight
0.15 nothing past ~86B is single-epoch -- equally true of the original directory. **If a 128B run is
published, the Setup sentence "no arm revisits a document" must be updated**; it is correct for every
32B arm. README in the shared dir carries this table.

---

### §19.8 (2026-09-22) `abl_S_134M_hopfield_surrogate` LAUNCHED — the arm that breaks the expert-form / router confound

Job **1862812**, 4 GPUs, `normal`/`grp_ebm`, host `p2-r07-n3`. Config
`configs/iclr_26/ablations/abl_S_134M_hopfield_surrogate.yml`.

**Why it exists.** Across the whole grid every hopfield arm uses the PROXY router and every w1w2 arm
the SURROGATE, so the w1w2 arm's -0.0108 nats advantage (§19.5) cannot be attributed. This one arm
fills the empty cell and isolates both variables:

| | proxy router | surrogate router |
|---|---|---|
| **hopfield** | 44.82 / 2.6544 | **abl_S (running)** |
| **w1w2** | does not exist | 45.01 / 2.6436 |

vs the w1w2 arm -> isolates EXPERT FORM at an identical router.
vs the hybrid baseline -> isolates the ROUTER at an identical expert form.

**Built from `cmix_134M_hyb_w1w2_sparse_surr_32B` with a 5-line diff**, every line accounted for:
`expert_kind w1w2->hopfield` (the variable), `intermediate_size 8192->16384` (forced: hopfield stores
ONE matrix per expert where w1w2 stores two, so this keeps it iso-param), `e_sign_override neg->pos`
(forced by pre-flight rule 8), plus `save_path` and the wandb name.

**Pre-flight, all verified BEFORE and AFTER launch:**
* datamix byte-identical to the cmix reference -- 0 diff lines (rule 1)
* params: total 134.13M / active 123.12M / FLOPwt 140.96M -- identical to the w1w2 arm to the byte,
  and within 0.09%/0.10%/0.54% of the hopfield+proxy baseline (rule 5 of the new-config gate)
* schedule sums to 122,070 (rule 1); no `load_args`; save_path did not pre-exist (rule 5)
* Sinkhorn rule 9: `persist_mu true`, `mu_iters 6` == the block's `layer_iterations` entry, 
  `repulsion_tensor_idx true`, no `cos_probe_interval`
* **live checks**: `DeviceMesh((pp=1, ddp=4, fsdp=1, tp=1), 'cuda', ...)` -- NOT cpu; 0 CUDA init
  errors; tokens/step derived from the log = **262,144** exactly; first step 50, i.e. a genuine
  fresh start
* registered in `watchdog_jobs.conf` at **4** GPUs (backup `.bak-20260922-ablS`). Registering after
  bsub is safe -- the watchdog adopts a same-named queued job instead of resubmitting.

**A QUEUE LESSON: `grp_ebm` is FREE again (4/32).** It submitted to `preemptable` with **735 jobs
pending ahead** and sat. `bmod -q normal -G grp_ebm 1862812` moved it and it started within a minute.
CLAUDE.md routes evals away from `grp_ebm` because it "routinely sits at 32/32" -- that was true while
the 400M arms ran, and is not true now. **Check `blimits` before assuming either queue is the fast
one.** Used `bmod`, NOT `bstop`, which invalidates `CUDA_VISIBLE_DEVICES` on a pending GPU job.

**ETAs at 0.40 s/step:** 8B crossing (step 30,517) **20:20Z tonight**, 16B 23:42Z, 32B **06:26Z Wed**.
By §18.22 the 8B loss read should settle the question; the full 32B is the publishable number.

**PROVISIONAL, RE-MEASURE LATER: hopfield may be ~49% slower in wall clock than w1w2 at equal
FLOPwt.** At 4 GPUs with the same surrogate router: w1w2 0.3304 s/step vs abl_S **0.4931** s/step,
both 140.96M FLOPwt. FLOPwt counts parameter-applications and says nothing about kernel efficiency;
hopfield runs one 16384-wide matrix with an elementwise gelu-and-square over that width where w1w2
runs two 8192-wide ones. **Do not quote this yet** -- abl_S was ~200 steps in with Triton autotuning
still active and two readings disagreed (0.397 over 5 steps, 0.493 over 20). Re-measure past a few
thousand steps, and report loss, Avg11 AND settled s/step together: an expert form that scores the
same for less wall clock is a win even at equal FLOPwt.

---

### §19.9 (2026-09-22) 1B SPEC: top_k is the wrong lever, 24 GPUs does not divide the budget, and `fused_experts` blocks multi-node

Asked to spec 1B runs at k/K = 4/64 and 6/64 on 24 GPUs, because the existing 2/64 arm "has too few
active". The active-parameter diagnosis is right; the proposed fix does not work.

**1. `top_k` barely moves active on this architecture** (`audit_config`, total is unchanged since the
same experts are stored):

| variant | total | active | act/tot | FLOPwt | compute |
|---|---|---|---|---|---|
| `8G4E` K=64 k=2 (existing) | 1002.07M | 279.32M | 27.9% | 279.32M | 1.00x |
| k=4 | 1002.07M | 302.63M | 30.2% | 302.63M | 1.08x |
| k=6 | 1002.07M | 325.95M | 32.5% | 325.95M | 1.17x |
| k=8 | 1002.07M | 349.26M | 34.9% | 349.26M | 1.25x |

**+17% active for +17% compute** -- it only slides along the same line. **8 of the 12 layers are
dense GPT and fully active regardless**, so the mixture modulates barely a third of the model.
Reaching ~500M active would need `k~21` (k/K = 0.33), at which point "sparse" is meaningless. This is
the same structural wall as §18.14: an active-parameter deficit cannot be fixed from the expert side
when dense layers dominate.

**2. A design that DOES fix it already exists, with its baseline.**
`configs/iclr_26/bharat/bharat_1B_18L_allMoE_{boltz,switch_baseline}.yml` -- 18 layers, hidden 1536,
EVERY block a mixture, K=32 top_k=8:

| design | total | active | act/tot |
|---|---|---|---|
| all-MoE 18L energy | 1022.42M | **512.81M** | **50.2%** |
| all-MoE 18L Switch (baseline) | 1004.43M | 494.82M | 49.3% |

**1.84x the active parameters of the current 1B arm**, and the pair is matched to 3.6% of active, so
the requested baseline is already specified. Prefer this over the top_k variants.

**3. 24 GPUs CANNOT hit the 1B tier's budget.** 524,288 tok/step = 128 sequences, and 128/24 =
**5.333** -- no integer `mbs x ga` exists. At 24 the global batch must be divisible by 24: 96 seqs
(-25%), 120 (-6.2%) or 144 (+12.5%). **Use 16 or 32 GPUs** (8 or 4 sequences per rank, exact).

**4. THE BLOCKER: `fused_experts: true` appears 18 times in the all-MoE config.** CLAUDE.md records it
as validated **single-node only** -- it WEDGES at 16 GPU / 2 nodes. Any launch at 16/24/32 GPUs risks a
silent wedge, so the pair needs `fused_experts: false` first, which forfeits the 1.61x fusion speedup.
(`repulsion_space: output` with `repulsion_subsample: 64` is already the multi-node-safe choice, so
that hazard does not apply.)

**5. Timing**, anchored on the measured 1.78 s/step at 8 GPUs / 279.32M FLOPwt, scaling s/step with
FLOPwt and assuming 80% multi-node efficiency (optimistic -- 1B multi-node is communication-heavy):

| design | 8 GPU | 24 GPU | 32 GPU |
|---|---|---|---|
| `8G4E` k=6 | 35.2 h | 14.7 h | ~11 h |
| all-MoE energy k=8 | 55.4 h | 23.1 h | ~17 h (≈27 h without fusion) |
| all-MoE Switch base | 53.5 h | 22.3 h | ~17 h (≈27 h without fusion) |

**Recommendation: the all-MoE pair with `fused_experts: false` at 32 GPUs; skip 4/64 and 6/64.**
Against the Sep 24 00:00Z deadline (~30 h at the time of writing) the pair does not fit sequentially
and needs 64 GPUs to run in parallel, so read them at their **8B milestone on loss** (§18.22:
Spearman 1.000 at 134M, 0.943 at 400M).

---

### §20.1 (2026-09-22) TWO CODE BUGS FOUND IN AUDIT. Both confirmed. Both affect published arms.

---

#### BUG A: `proxy_init: svd` has NEVER taken effect. It fails under FSDP and the failure is swallowed.

`_svd_refit_proxy()` (`energy_ff.py:1165`) raises on its first write and is caught by a bare
`except Exception`, which logs a warning and sets `_svd_done = True` so it never retries:

```
RuntimeError('aten.copy_.default got mixed torch.Tensor and DTensor, need to convert all
torch.Tensor to DTensor before calling distributed operators!')
```

**893 occurrences across 36 logs.** Cause: `proxy_V` / `proxy_B` are `nn.Parameter`s that FSDP shards
into DTensors, while the SVD factors (`Vh`, `US`) are plain tensors; `proxy_V.data[k].copy_(plain)`
is illegal. The `try/except` was added to stop a warm start killing a long run, and it did its job
too well -- it turned a hard failure into a silent no-op.

**Nine arms specify `proxy_init: svd` and every one of them actually ran with RANDOM proxy init:**
`cmix_134M_hybrid_32B_sparse` (the BASELINE), `cmix_134M_sandwich_32B_sparse`,
`cmix_134M_pure_32B_sparse`, `cmix_400M_hybrid_sparse`, `cmix_400M_sandwich_sparse`,
`abl_C_134M_1G1x6E1G_isototal`, `abl_R_134M_hyb_rnorm_none`, `abl_G_400M_6G1x6E1x6E`,
`abl_H_400M_6G6E_deep`.

**Size of the loss.** The function's own docstring measures top-2 router agreement against the exact
energies on a trained 1B checkpoint: **0.5541 SVD / 0.0378 random** -- and distillation is supposed
to continue from the SVD point. So a ~15x better starting point was specified and silently discarded
on every sparse(proxy) arm we have published.

**The originally-reported concern is NOT what happens.** The audit flagged "re-initialised from SVD at
every job resume". `_svd_done` is a plain Python attribute, not a registered buffer, so it does reset
to `False` in every new process, and the refit IS re-triggered on the
`_sparse_active: False -> True` transition that a resume past `sparse_start_step` always produces.
But the refit dies on its first `copy_`, so **nothing is overwritten and the loaded distilled proxy
survives**. The resume path is accidentally safe. Fixing the DTensor bug WITHOUT also persisting
`_svd_done` would turn this latent bug into a live one -- every resume would then clobber the
distilled proxy with a fresh SVD. **Both must be fixed together.**

Exposure requires `sparse_forward: true` AND `sparse_start_step > 0` AND `proxy_init: svd`. Arms with
`proxy_init: random` (`cmix_134M_hyb_w1w2_*`, `abl_I`, `cmix1B_12L_gptDense_32B`) are unaffected.

---

#### BUG B: every auxiliary loss is silently multiplied by 0.001, and the Switch baselines use 10x more.

`lm_engine/model_wrapper/pretraining.py:543`:
```python
return lm_loss + router_aux_loss_coef * aux_loss
```
`router_aux_loss_coef` defaults to **0.001** in `CommonConfig` (`config/__init__.py:145`) and is
**unset in every energy config**. Everything routed through `add_aux_loss` -- expert repulsion, proxy
distillation, surrogate distillation, the legacy top-k load-balance loss -- is scaled by it.

| config field | nominal | EFFECTIVE |
|---|---|---|
| `repulsion_coef` | 0.1 | **1e-4** |
| `proxy_loss_coef` | 0.01 | **1e-5** |

**The asymmetry nobody noticed:** the only two configs in the tree that set the field are
`configs/iclr_moebase/iclr_switch_K16_top2{,_shared}.yml`, both at **0.01** -- **10x** the default
the energy arms silently use. So in the Switch-vs-energy comparisons the baseline's load-balance loss
carries ten times the weight of the energy arms' auxiliary losses. Different losses, but an
unintended and undocumented 10x asymmetry between arms we compare directly.

**This plausibly explains §19.5's "every knob is inert".** Repulsion at an effective 1e-4 would do
very little whatever its nominal value, so that whole sweep may have been measuring a disabled
mechanism rather than a mechanism without effect. **Re-run the knob sweep before quoting §19.5.**

---

**`abl_P_134M_pure_223_isoall` (job 1865020) was launched WITH BOTH BUGS INTACT, deliberately.** It
exists to be compared against the 134M hybrid / dense / Switch arms, and fixing it alone would make
it the only arm with different aux weighting and proxy init. Comparability beats correctness for that
one arm. Any fix must therefore be opt-in, or the live arms' configs must be pinned to their actual
behaviour so a requeue reproduces it.

---

### §20.2 (2026-09-22) BOTH FIXES VERIFIED IN PRODUCTION, and Bug B's suppression proved arithmetically

#### Bug A (`proxy_init: svd` no-op) — fixed and validated twice

**Fix, in `energy_ff.py`:** two changes that had to land together.
1. `_copy_full_into(param, full_new)` + `_gather_full(param)` — build FULL replacement tensors for
   `proxy_V` / `proxy_B` / `proxy_scale` / `proxy_bias`, then write each parameter ONCE, going through
   `torch.distributed.tensor.distribute_tensor` when the target is a sharded DTensor. The old code
   wrote per-expert with `param.data[k].copy_(plain)`, which is the illegal mixed Tensor/DTensor op.
2. **`_svd_done_buf` registered as a PERSISTENT buffer**, and the guard now reads it. Without this,
   fixing (1) would have CREATED the bug the audit feared: `_svd_done` is a plain attribute that
   resets per process, so a resume past `sparse_start_step` re-fires the refit and would now
   successfully overwrite a distilled proxy with a fresh SVD.

**Validation 1** — dedicated 400-step 2-GPU run (`_svdfix_test.yml`, job 1865216) crossing
`sparse_start_step=300`: `aten.copy_.default got mixed` **0** (was 893 across old logs),
`proxy SVD refit skipped` **0**, reached step 400, `lm_loss` 8.3898 -> 5.6290, no OOM, no assertions.

**Validation 2, the one that matters** — the five live knob arms at 4-GPU FSDP, all past step 5,000
with `proxy_init: svd`:

| arm | step | refit-skipped | DTensor errors |
|---|---|---|---|
| `knob_rep_off` | 5370 | 0 | 0 |
| `knob_rep_10x` | 5020 | 0 | 0 |
| `knob_rnorm_none` | 5150 | 0 | 0 |
| `knob_tau035` | 5050 | 0 | 0 |
| `knob_proxy_10x` | 5060 | 0 | 0 |

**Still untested:** (a) that the refit's OUTPUT is numerically the right factorisation -- only that it
no longer throws; proving it needs comparing fitted `proxy_V`/`proxy_B` against a direct SVD of `W`.
(b) the persistent-buffer path, which only runs on an actual requeue.

#### Bug B (aux losses x0.001) — suppression proved, and the fix verified by arithmetic

Not fixed by changing the global default (that would make every published arm unreproducible from its
config). Instead set **`router_aux_loss_coef: 1.0` per-arm** in the six knob configs.

**The proof it is in effect:** with the coefficient at 1.0, `train-loss - train-lm_loss` must equal
`train-aux_loss` exactly. Measured:

| arm | train-loss | lm_loss | gap | aux_loss |
|---|---|---|---|---|
| `knob_rep_off` | 2.7938 | 2.7872 | **0.0066** | **0.0066** |
| `knob_rep_10x` | 2.8379 | 2.8273 | **0.0106** | **0.0106** |

Under the old 0.001 default the gap would have been 6.6e-6 -- invisible in the logs, which is exactly
why this went unnoticed for the whole project. **The auxiliary losses now reach the optimiser at
their nominal strength for the first time.**

**A structural read that falls out of it:** `knob_rep_off` sets `repulsion_coef: 0.0` yet still shows
`aux_loss = 0.0066`, so that component is the PROXY DISTILLATION loss; repulsion at `coef 1.0` adds
roughly the remaining 0.004. The two mechanisms are live and separable, which is what makes the sweep
interpretable at all.

#### A cache race I caused, worth not repeating

Launching all eight jobs at once against the SAME `data_cache_path` made several build the same
Megatron index concurrently; two read a `.npy` mid-write and died with
`ValueError: EOF: reading magic string, expected 6 bytes got 0` (`knob_ctrl_aux1`, `abl_S`). The
cache is keyed on (mix, num_samples, seed), so arms with the same step count contend. **Build the
index once -- or stagger the launches -- before fanning out onto a shared cache.** The cache is now
complete (126 files, none truncated) and the relaunches read it cleanly.

#### Reading discipline for this sweep

The arms are at DIFFERENT steps while running (5,020-5,370 above), and a cross-step loss comparison is
meaningless. Wait for the matched 8B milestone. Also: the control differs from the published baseline
by **both fixes AND the aux coefficient at once**, so any gap there is not attributable to a single
cause -- separating them needs one more arm with the SVD fix but the old 0.001 coefficient.

---

### §20.3 (2026-09-22) IS THE SPARSITY REAL? Yes in training. NO at eval — and the evals never touch the proxy.

Asked whether `abl_P` (and the `pure` 32B arm it inherits from) is genuinely sparse, or whether
`total == active`.

**`total != active`. The parameter accounting is correct.** `_route` (`energy_ff.py:1358-1368`) applies
top-k masking on BOTH forward paths:
```python
_, topk_idx = sel_logits.topk(self.top_k, dim=-1)
mask = torch.zeros_like(p, dtype=torch.bool); mask.scatter_(-1, topk_idx, True)
p = p * mask                     # only top_k experts contribute to the output
```
So only 2 of 32 experts affect the output whichever path runs, and `active = 123.13M` for `abl_P` is
right. **Both arms DO train sparse from step 300.**

**What differs between the paths:**

| | selection | skipped experts | compute |
|---|---|---|---|
| training, step >= 300 (`_forward_sparse`) | **proxy** top-k | truly not evaluated | sparse |
| eval (`_forward_fused`) | **exact** energies (oracle) | all K evaluated then masked | dense |

**THE MISMATCH.** `set_training_step` is called ONLY from `lm_engine/pretrain.py:403` -- the training
loop. `_sparse_active` is a plain Python bool initialised to
`bool(sparse_forward) and sparse_start_step <= 0`, which is **False** whenever
`sparse_start_step > 0`. An eval process constructs the module fresh and never calls
`set_training_step`, so `_sparse_active` stays False and the eval takes the NON-sparse path.

Consequences, for every sparse(proxy) arm we have published:
1. **Reported Avg11 / ppl / MMLU / GSM8K use ORACLE routing, not the proxy the arm trained with.**
   They are an upper bound on what the deployed sparse model would score.
2. **Proxy error never penalises the eval numbers.** This is why proxy-quality knobs could not show up
   in Avg11 -- the evals never ran the proxy. It also means Bug A and Bug B degraded TRAINING
   DYNAMICS only, not the measured accuracies.
3. **Eval compute is dense** even though the parameter accounting is sparse. A throughput claim
   measured at eval would not reflect the sparse path.

Whether to call this a bug depends on the claim being made. For "does energy routing learn a good
mixture" the oracle eval is arguably the right measurement. For "sparse energy routing is deployable
at k/K cost" it is not, and the paper should say which it means. Note `abl_P` has
`sparse_candidates: 2` equal to `top_k: 2`, so the proxy nominates exactly two candidates with no
re-ranking slack -- the sparse path's selection IS the proxy's, with no exact-energy correction.

**To evaluate on the sparse path** one of these is needed: set `sparse_start_step: 0` for the eval
load (then `_sparse_active` is True at construction), or make `_sparse_active` a persistent buffer, or
have the eval harness call `set_training_step(big)` after load.

---

## 21. SPARSE EVAL WAS NEVER SPARSE — every published number used ORACLE routing (2026-09-22)

### 21.1 The bug

`BoltzmannMoEFFEnergy.__init__` (`energy_ff.py:923`) sets

```python
self._sparse_active = bool(sparse_forward) and self.sparse_start_step <= 0
```

and the **only** thing in the tree that ever flips it is `set_training_step()`, called from exactly
one place: the training loop, `lm_engine/pretrain.py:403`. Nothing in `from_pretrained`, `.eval()`,
`.to()`, or the eval harness touches it.

Every checkpoint is trained with `sparse_start_step > 0` — it has to be, because the proxy is
distilled against the exact all-K routing distribution that the sparse path never computes. So
**every checkpoint reloads for evaluation with `_sparse_active = False` and is evaluated on the
DENSE all-K path with EXACT (oracle) routing.** The proxy / surrogate router — whose approximation
error *is* the sparsity claim — never ran in a single published eval.

Why it survived this long:

* it fails **silently and in the safe direction**: oracle routing is the best case, so nothing
  crashes and no number looks anomalous;
* `energy_ff.py:952` states outright that "the main use of `sparse_forward` is EVALUATING an
  already-trained checkpoint", so the intent was documented and only the wiring was absent;
* `scripts/auto_eval_on_finish.sh:101` had already **observed the symptom** — "a checkpoint saved
  with `sparse_start_step>0` reloads with `_sparse_active` FALSE, so eval runs the DENSE all-K path
  and single-block arms OOM at batch 4" — and worked around the resulting OOM with `batch_size 1`
  instead of the cause.

**Scope: 28 checkpoint directories carry `sparse_forward` + `sparse_start_step > 0`; 25 of them have
stored evals.** That covers every sparse row of Table 1 and Table 6, both 400M arms, `abl_H`,
`abl_I`, `abl_R`, `abl_T`, and the 1B `gptDense` arm.

### 21.2 The fix — config patch, no model-code change, validated

`scripts/make_sparse_eval_dir.sh <unsharded_dir> <dst>` builds a parallel eval directory whose
`config.json` says `sparse_start_step: 0`, with the weights **hard linked** (same inode → zero
space, cannot drift). `sparse_start_step <= 0` makes `_sparse_active` True at construction *and*
makes `set_training_step` a no-op, so the proxy loaded from the checkpoint is used verbatim and is
never re-fitted. **The original directory is untouched**, so the dense/oracle numbers survive for
the oracle-vs-sparse comparison the paper needs.

**This is inference-only. `sparse_start_step: 300` remains correct and unchanged in every training
config** — the dense warmup is load-bearing. Verified: no tracked config YAML modified; the only
`"sparse_start_step": 0` values written are inside `sparseeval_*` dirs; and no `load_args` anywhere
points at an `unsharded_*`/`sparseeval_*` dir, because those are eval artifacts and training resumes
from the sharded `save_path`.

**Validated by probe job 1866475** on `cmix_134M_hybrid_32B_sparse/sparseeval_step122070`, 4 cheap
tasks, same weights / harness / code, one integer changed. Deterministic likelihood evals cannot move
unless the forward path changed, and all four moved:

Accuracy fractions (0–1), higher is better; `sciq`/`piqa` are `acc_norm`, `copa`/`winogrande` `acc`.
Same checkpoint, step 122070, 32B tokens.

| task | oracle (dense all-K) | proxy-sparse | delta |
|---|---|---|---|
| copa | 0.6400 | 0.6300 | −0.0100 |
| sciq | 0.7180 | 0.7270 | **+0.0090** |
| piqa | 0.6262 | 0.6251 | −0.0011 |
| winogrande | 0.5217 | 0.5146 | −0.0071 |

No traceback, no OOM, no assert. Mean −0.23pp over the four — the proxy looks close to oracle, which
is the direction the paper wants, but **the 11-task numbers are what decide it.**

A second route, for callers holding a live model that cannot restage a directory, is the new
**`lm_engine/hf_models/modeling_utils/mlp_blocks/sparse_eval.py`**:
`activate_sparse_inference(model)`, `sparse_inference_report(model)`, `assert_sparse_active(model)`.
It is a **new file imported by nothing in the training path** (deliberately: making `_sparse_active`
a persistent buffer, or flipping it in `train(False)`, would change training behaviour and break
strict loading of existing checkpoints). `assert_sparse_active` is the guard to put in any future
eval so a dense-by-accident run can never again pass for a sparse one.

### 21.3 Every wave-A arm really does prune

Read from the patched configs. `K` = experts, `k` = `top_k` applied, `p` = candidates the cheap
router nominates, `r` = proxy rank. `p < K` is what makes the eval genuinely sparse.

| arm | K | k | p | r | router |
|---|---|---|---|---|---|
| 134M hybrid / pure / sandwich | 16 | 2 | 2 | 16 | rank-16 subspace proxy |
| 134M w1w2 surrogate, `abl_I` | 16 | 2 | 4 | – | surrogate MLP head (d→256→16) |
| 400M hybrid / sandwich, `abl_H` | 32 | 2 | 2 | 16 | rank-16 subspace proxy |
| 1B `gptDense` | 64 | 2 | 2 | 16 | rank-16 subspace proxy |

The surrogate arms initially looked misconfigured (`proxy_rank: 0`, `surrogate_coef: 0.0`,
`use_surrogate: False` on an arm named `_surr_`). They are correct: `surrogate_replaces_proxy: true`
forces `proxy_rank → 1` internally before `super().__init__` and **overrides `_proxy_energies` with
the MLP head**, which supplies the nomination, the zscore moments and the denominator tail while the
exact energies of the nominated `p` still re-rank to the final top-k. `use_surrogate: False` is
*required* there and mutually exclusive with the flag — setting it would let the head set mixture
weights, i.e. exactly the Switch-style gate this design exists not to be. The head is trained via
`proxy_loss_coef: 0.01`, not `surrogate_coef`.

### 21.4 Bug B makes today's sparse penalty an UPPER BOUND

Every one of these routers is distilled through `add_aux_loss`, so its loss is multiplied by
`router_aux_loss_coef`, whose default is **0.001** and which no energy config sets (§20). Effective
distillation weight is therefore `proxy_loss_coef 0.01 × 0.001 = 1e-5`. **The proxy and surrogate
routers in every evaluated arm were trained with a ~1000× attenuated distillation loss.** So
whatever sparse-vs-oracle gap the wave-A evals report is a *pessimistic* bound on what a correctly
weighted proxy would deliver — and it also explains cleanly why the proxy knobs could never move
Avg11: the proxy was barely trained *and* was bypassed at eval.

### 21.5 Operational notes

* **HF Hub 429.** Submitting 18 evals at once made each pull MMLU's 57 subject configs from the Hub
  and got our IP rate-limited; 5 jobs died in 3 minutes with `429 Client Error`. Every dataset the 13
  tasks need is already in `~/.cache/huggingface`, so the fan-out now exports
  `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` and staggers submissions by 4 s. **Any future eval burst
  must run offline.**
* **Milestones are deferred.** `scripts/fanout_sparse_evals.sh` splits targets into wave **A**
  (Table 1 + Table 6 finals — the only thing that matters for the paper), wave **B** (short 8B
  ablation finals), wave **C** (intermediate milestones, **do not run until every wave-A final is
  redone**).
* **Bug A had a second copy.** `energy_ff_w1w2_sparse.py:653` repeated the plain-`_svd_done` defect
  the base class was fixed for; patched the same way (read the persistent `_svd_done_buf`, and set it
  on all three settled exits). It could not have fired on the shipped surrogate arms, which have
  `proxy_rank: 0` and return early, but it would on any future w1w2 arm with a real proxy.

### 21.6 A SECOND, independent instance — and this one is in the MAIN paper table

The §21.1 gate bug affects checkpoints whose config sets `sparse_start_step > 0`. Arms trained
before that field existed omit it, it defaults to 0, and they DO activate the sparse path at load.
So the older `iclr_*` grid is not hit by the gate bug. It is hit by a different one.

`tab:cost` in `sec/experiments.tex` has the row **"Boltzmann, $K{=}32$, proxy router ... 44.58"**.
That 44.58 traces to `results/iclr_sink/iclr_hop_K32_top2_sink/unsharded/`, whose config has
`sparse_forward` **absent** and `proxy_rank: 0` — a fully DENSE, no-proxy export. The arm's actual
proxy export, `unsharded_sparse_r16m512` (`sparse_forward: true`, `proxy_rank: 16`, p=2 of K=32),
carries **zero evals**. So the accuracy on a row explicitly labelled *proxy router* is the accuracy
of the model WITHOUT the proxy, while the MACs and cost columns describe the proxy configuration.
That is a composite claim assembled from two different models.

A scan for the same pattern — a sparse export with no evals beside a dense export that has them —
finds it in **three** arms, all of them `iclr_sink`, i.e. the family the paper headlines:

| arm | sparse export (no evals) | dense export used for the published number |
|---|---|---|
| `iclr_hop_K32_top2_sink` | `unsharded_sparse_r16m512` (r=16, p=2/32) | `unsharded` → Avg11 44.58 |
| `iclr_big_hop_sandwich_sink` | `unsharded_sparse_r16m512` (r=16, p=2/16) | `unsharded_mucal` |
| `pure_hop_T12_sink` | `unsharded_sparse_r16m512`, `unsharded_sparse_cal` | `unsharded_mucal2` → 40.96 |

Evals submitted for all three sparse exports (jobs 1867516/17, 1867523/24, 1867525/26). These need
no config patch: their `sparse_start_step` is absent, so `_sparse_active` is already True.

**Net scope of "the sparsity claim was never measured":** 25 checkpoints via the gate bug (§21.1)
plus 3 via an unevaluated sparse export. The two mechanisms are different but the consequence is
identical, and together they cover **every sparse number in the paper**.

---

## 22. KNOB SWEEP, POST-BUG-B — expert repulsion HURTS, monotonically (2026-09-22)

Six 134M hybrid arms, identical but for ONE knob each (verified by config diff: every pair differs
only in the knob line, the `save_path` and the wandb `name`). All six set
`router_aux_loss_coef: 1.0`, i.e. this is the FIRST sweep run with Bug B corrected — the auxiliary
losses actually reach the objective at their stated coefficients. `knob_ctrl_aux1` is the control:
same corrected aux weight, all knobs at their shipped values.

Read on `train-lm_loss` at MATCHED steps (the control started later, so its own final loss is not
comparable). Median over a +-500-step window, ~101 samples, SE ~0.0026 nats.

**Values are `train-lm_loss` in NATS, LOWER IS BETTER.** Rank 1 = best.

| step | 1st | 2nd | 3rd | 4th | 5th | 6th |
|---|---|---|---|---|---|---|
| 10000 | rep_off 2.6194 | ctrl 2.6242 | proxy10x 2.6271 | tau035 2.6277 | rnorm_none 2.6398 | rep_10x 2.6449 |
| 14000 | rep_off 2.5516 | tau035 2.5574 | ctrl 2.5596 | proxy10x 2.5607 | rnorm_none 2.5698 | rep_10x 2.5807 |
| 18000 | rep_off 2.4951 | tau035 2.4997 | ctrl 2.5011 | proxy10x 2.5068 | rnorm_none 2.5110 | rep_10x 2.5246 |
| 20000 | rep_off 2.4705 | ctrl 2.4764 | tau035 2.4766 | rnorm_none 2.4852 | proxy10x 2.4880 | rep_10x 2.5031 |

**`rep_off` is rank 1 at all four steps and `rep_10x` is rank 6 at all four.** The repulsion
dose-response is monotone at every step, deltas against the control in nats:

| step | `repulsion_coef` 0.0 | 0.1 (control) | 1.0 |
|---|---|---|---|
| 10000 | **−0.0048** | 0.0000 | +0.0207 |
| 14000 | **−0.0080** | 0.0000 | +0.0211 |
| 18000 | **−0.0060** | 0.0000 | +0.0235 |
| 20000 | **−0.0059** | 0.0000 | +0.0267 |

Off is best, 0.1 is worse, 1.0 is worst, at every step — and the 1.0 penalty GROWS with training
(+0.021 → +0.027), which is what a genuine regulariser mismatch looks like rather than noise.

### What this means, and what it does NOT justify

1. **Expert repulsion is not earning its place.** Turning it off is both better and simpler (one
   fewer loss term and hyperparameter). Adopt `repulsion_coef: 0.0` for new arms.
2. **Bug B was accidentally PROTECTIVE.** The published arms ran repulsion at an effective
   `0.1 × 0.001 = 1e-4` — essentially off, which this sweep says is the best setting. So naively
   "fixing" Bug B by setting `router_aux_loss_coef: 1.0` while leaving `repulsion_coef: 0.1` would
   make results slightly WORSE (−0.006 nats), and leaving repulsion at 1.0 would cost 0.027 nats.
   **The correct fix is to raise the coefficient AND turn repulsion off**, not one without the other.
   This also retires the worry that the published arms were damaged by Bug B on the repulsion term:
   on this evidence they were not, they were helped.
3. **It is NOT a reason to relaunch a 32B arm.** −0.006 nats maps through the 134M cross-arm slope
   (−8.6 pp/nat) to about **+0.05pp Avg11**, far inside the 0.73pp seed range. With ~25 h to the
   deadline the expected gain does not pay for a 32B run, so no new hybrid was launched.

### The other three knobs

* **`temperature: 0.35`** — tied with the control (−0.0014 to +0.0002 across steps; rank moves
  between 2nd and 3rd). No effect worth acting on.
* **`proxy_loss_coef: 0.1` (10x)** — slightly WORSE (+0.0029 to +0.0116). A stronger proxy
  distillation does not help the language loss, which is expected: the proxy is off the main
  gradient path and only affects *selection*. Its real test is the sparse-eval gap (§21), not loss.
* **`routing_norm: none`** — WORSE on loss (+0.0088 to +0.0156) at every step. Note this arm's
  32B Avg11 went the OTHER way (44.92 oracle, and **45.43 sparse**, both above the hybrid's 44.82).
  That is the documented loss/Avg11 sign-flip for this exact knob (§18) — which is why the 8B
  decision rule is stated for LOSS RANKING ONLY and never for Avg11.

**Caveat:** read at steps 10000–20000 (2.6–5.2B tokens), not at the 8B milestone, because
`knob_ctrl_aux1` was relaunched after a data-cache race and is ~7k steps behind. The ordering is
stable across all four reads, so the conclusion does not depend on the final read; the 30518-step
(8B) confirmation is still pending.

### 21.7 THE RESULT: sparsity is free IF the model trained with it, and catastrophic if grafted on

With 13 of 16 exports re-evaluated, the arms split into two populations that must never be pooled.
The mean over all 16 (−0.59pp) is meaningless; it is an average over two different experiments.

**Columns: Avg11 = 11-task accuracy mean in PERCENTAGE POINTS, higher better. ppl = wikitext
WORD-level perplexity, LOWER better. ORACLE = dense all-K exact routing (what every published
number was); SPARSE = the cheap router actually selecting. Each arm at its own final checkpoint.**

#### Population A — trained WITH the sparse path (`sparse_start_step: 300`, `proxy_loss_coef > 0`)

The proxy was distilled during training AND the expert weights co-adapted to being selected by it.

| arm | K | p/K | Avg11 orc | Avg11 sps | ΔAvg11 | ppl orc | ppl sps |
|---|---|---|---|---|---|---|---|
| 134M sandwich | 16 | 2/16 | 43.45 | **44.34** | **+0.89** | 40.98 | 41.79 |
| 134M rnorm=none | 16 | 2/16 | 44.92 | **45.43** | **+0.51** | 42.36 | 42.79 |
| 134M rnorm=none @32k | 16 | 2/16 | 42.38 | 42.57 | +0.20 | 58.25 | 58.51 |
| 134M w1w2 unc proj | 16 | 4/16 | 44.32 | 44.37 | +0.06 | 41.09 | 40.74 |
| 400M sandwich | 32 | 2/32 | 44.52 | 44.52 | +0.00 | 41.13 | 43.05 |
| 134M τ=2.0 @32k | 16 | 2/16 | 42.20 | 42.18 | −0.02 | 57.35 | 59.71 |
| 134M w1w2 surrogate | 16 | 4/16 | 45.01 | 44.93 | −0.08 | 40.70 | 40.85 |
| 134M hybrid | 16 | 2/16 | 44.82 | 44.67 | −0.15 | 41.06 | 41.82 |
| 134M τ=0.35 @32k | 16 | 2/16 | 42.87 | 42.61 | −0.26 | 56.78 | 59.61 |
| 1B gptDense | 64 | 2/64 | 47.79 | 47.42 | −0.37 | 29.46 | 30.13 |
| 400M `abl_H` 6G6E | 32 | 2/32 | 46.92 | 45.97 | −0.96 | 30.14 | 32.22 |

**mean ΔAvg11 = −0.02pp, range −0.96 to +0.89, four arms IMPROVE.** Selecting 2 of 64 experts with a
rank-16 proxy costs 0.37pp at 1B; selecting 2 of 16 costs 0.15pp at 134M. Perplexity degrades
slightly and consistently (+0.1 to +2.1), which is the honest signature of a small approximation —
the Avg11 scatter around zero is task-level noise on top of it.

#### Population B — proxy GRAFTED post-hoc onto a DENSE-trained arm (`proxy_loss_coef: 0.0`)

These `iclr_sink` exports were built by fitting a proxy offline to an arm that trained with dense
soft routing. The proxy was never distilled and the weights never co-adapted.

| arm | K | p/K | Avg11 orc | Avg11 sps | ΔAvg11 | ppl orc | ppl sps |
|---|---|---|---|---|---|---|---|
| `iclr_hop_K32_top2_sink` | 32 | 2/32 | 44.58 | 43.04 | **−1.53** | 40.65 | **69.05** |
| `iclr_big_hop_sandwich_sink` | 16 | 2/16 | 43.36 | 37.39 | **−5.97** | 44.62 | **269.94** |

**−1.5 to −6.0pp, with perplexity 1.7× to 6.1× worse.** A word-perplexity of 269.94 against 44.62 is
not a degraded model, it is a broken one.

#### Why this matters more than the headline number

1. **Co-adaptation is the mechanism, not the proxy's raw accuracy.** The same rank-16 subspace proxy
   at the same p/K is free in population A and destructive in population B. What differs is whether
   the model trained through it. This is a *positive* result for the method as shipped and a sharp
   limit on post-hoc sparsification.
2. **It kills `tab:cost` as written.** That table's "proxy router" row IS population B — the
   `iclr_hop_K32_top2_sink` post-hoc export. The draft claims the proxy buys parity on arithmetic
   "for $0.25$pp of accuracy and a measured $+0.13$ perplexity". The truth for that arm is
   **−1.53pp and +28.4 perplexity**. Worse, the $0.25$pp in the prose is not the proxy's cost at all:
   $44.83 - 44.58 = 0.25$ is the *Switch gap*, while the table gives the proxy row the SAME $44.58$
   as the exact row — so the table asserts the proxy is free and the prose asserts it costs
   $0.25$pp, and neither had been measured.
3. **The fix is available and is better for us:** report the population-A arms, where the proxy is
   genuinely near-free, and state the post-hoc result as the limitation it is. That is both honest
   and a stronger claim than the current text.

**Pending:** `measure_stored_proxy_agreement.py` (jobs 1868229/30/31) measures the top-k SET overlap
between each STORED proxy and the exact all-K energies — the existing calibration script fits a
FRESH proxy and so cannot answer this. If population B's agreement is near chance ($k/K$), those
exports were simply never fitted and the right correction to `tab:cost` is to fit one properly; if
agreement is high and accuracy still collapses, post-hoc grafting genuinely fails. **Do not state a
cause in the paper until that lands.**

### 22.1 The 8B read, and what it says about Bug B

Five arms reached the 8B milestone (step 30510 = 7.997B tokens). `knob_ctrl_aux1` is ~6.5k steps
behind after its data-cache relaunch, so the control comparison is quoted at step 23000, the highest
step all six passed.

**`train-lm_loss` in NATS, LOWER IS BETTER.** Median over the window, ~42 samples at 8B (SE 0.0045),
~101 at step 23000 (SE 0.0025).

| rank | arm @ 8B (step 30400) | loss | | arm @ step 23000 | loss | vs control |
|---|---|---|---|---|---|---|
| 1 | **`repulsion_coef: 0.0`** | **2.3590** | | **`repulsion_coef: 0.0`** | **2.4221** | **−0.0081** |
| 2 | `routing_norm: none` | 2.3692 | | control (shipped knobs) | 2.4302 | 0.0000 |
| 3 | `temperature: 0.35` | 2.3700 | | `temperature: 0.35` | 2.4316 | +0.0014 |
| 4 | `repulsion_coef: 1.0` | 2.3926 | | `routing_norm: none` | 2.4349 | +0.0047 |
| 5 | `proxy_loss_coef: 0.1` | 2.4273 | | `repulsion_coef: 1.0` | 2.4543 | +0.0241 |
| 6 | — | — | | `proxy_loss_coef: 0.1` | 2.4547 | +0.0245 |

**`repulsion_coef: 0.0` is rank 1 at every one of the six reads** (steps 10k, 14k, 18k, 20k, 23k,
30.4k), by 0.006–0.010 nats against the control. That is the single robust result of the sweep.

The middle ranks shuffle between reads (`routing_norm: none` is 4th at 23k and 2nd at 8B), so only
the extremes are trustworthy: repulsion off is best, and **`proxy_loss_coef: 0.1` is worst at 8B**,
0.068 nats behind.

**This revises §21.4's framing of Bug B.** Both knobs that make an auxiliary loss STRONGER make the
LM loss WORSE, now that `router_aux_loss_coef: 1.0` lets them through at full strength. Combined with
§21.7 — where the shipped arms' proxies, distilled at an effective $10^{-5}$, already make sparsity
free — the conclusion is that **Bug B was largely benign and in part protective**:

* on repulsion it was clearly protective (off is best, and the shipped arms were effectively off);
* on proxy distillation it cost nothing measurable, because the proxies it under-trained are already
  good enough that population A loses 0.02pp on average to sparse selection.

So the §21.4 caveat that the sparse gaps are "an upper bound" should be kept as a statement of
*direction* but stripped of any suggestion that a correctly weighted proxy would be much better:
the control here runs proxy distillation 1000x stronger than the published arms and is not better,
and 10x stronger again is decisively worse.

**Recommended settings for new arms:** `repulsion_coef: 0.0`, `proxy_loss_coef: 0.01`, and either
`router_aux_loss_coef` left at its 1e-3 default or raised to 1.0 — the two are indistinguishable on
this evidence once repulsion is off. **Do not raise `proxy_loss_coef`.**

**Still not a reason to relaunch a 32B arm:** −0.008 nats maps through the 134M cross-arm slope
(−8.6 pp/nat) to ~+0.07pp Avg11, well inside the 0.73pp seed range.

### 21.8 Proxy AGREEMENT does not predict accuracy loss — co-adaptation does

`scripts/measure_stored_proxy_agreement.py` measures the top-$k$ SET overlap between the proxy AS
STORED IN THE FILE and the exact all-$K$ energies. (The existing
`calibrate_proxy_router_20260916.py` fits a FRESH proxy and reports what a good fit could achieve,
which cannot audit an existing export.) Chance is $k/K$.

| arm | population | ΔAvg11 | stored-proxy agreement | chance |
|---|---|---|---|---|
| `iclr_hop_K32_top2_sink` | B, post-hoc graft | **−1.53** | **0.666** | 0.0625 |
| `cmix_400M_sandwich` | A, trained sparse | **+0.00** | **0.429** | 0.0625 |

**The arm with the WORSE proxy loses NOTHING, and the arm with the BETTER proxy loses 1.53pp.**
Both are far above chance, so neither export is unfitted — the "never fitted" hypothesis in §21.7 is
**refused**.

The implication is the interesting one. Agreement measures whether the cheap router picks the experts
the *exact energy ranking* would pick. In an arm trained through the proxy, that is the wrong
question: the experts co-adapted to being selected by this router, so its "disagreements" are not
errors — the exact-energy ranking has stopped being the gold standard. In an arm trained densely and
sparsified afterwards, the exact ranking IS the standard the weights were fitted to, so every
disagreement is a real substitution and costs accuracy.

**This weakens a paper claim we currently lean on.** `sec/experiments.tex` around line 441 uses
nomination recall (0.47 for a linear head against 0.93 for a rank-$r$ subspace proxy) as the figure
of merit for a selector. On this evidence recall is not predictive of end-task cost, and a recall
number without stating whether the arm trained through the selector is not interpretable. Either
report the end-task delta instead, or state recall strictly as a property of the selector rather than
as evidence about accuracy.

**Caveat on the measurement, stated because it bounds the claim.** The collection is forced onto the
DENSE path (`_sparse_active = False`) because `Collector` hooks `_route`, which `_forward_sparse`
never calls — it inlines `_logits_raw`. So agreement is computed on the hidden states a DENSE forward
produces. For a co-adapted arm the sparse forward visits a somewhat different activation
distribution, so 0.429 is "agreement on dense activations", not exactly the agreement the deployed
model experiences. It is the right comparison across arms (identical protocol for both) but should
not be quoted as the deployed selector's fidelity.

---

## 23. THE RESCUED DATASET COPY IS A BIASED PREFIX — no loss is comparable across copies (2026-09-22)

### 23.1 What was copied

The 2026-09-22 rescue (owners were deleting the source) produced
`/proj/dmfexp/datasets-shared/granite-4-cmix-subset/`:

| dataset | copied | fraction | note |
|---|---|---|---|
| `web-nemotron-cc-hq-p2_0` | 200,000,000,600 B = **50B tokens** | **18.6%** | byte PREFIX |
| `web-nemotron-cc-hq-p2_1` | 200,000,002,916 B = **50B tokens** | **18.7%** | byte PREFIX |
| `megamath-web-pro_0` | 52,045,089,268 B | **100%** | complete |
| `finemath-3plus-rewritten_0` | 84,474,794,208 B | **100%** | complete |

The full corpus is 268.7B tokens per web shard. (The README in `full-corpus-indices/` still says the
subsets are 20B tokens — that is STALE, `extend_web_subsets_to_50B.sh` grew them to 50B.)

### 23.2 The prefix is NOT an unbiased sample: 0.47 nats

`scripts/test_subset_prefix_bias_20260922.py` scores a checkpoint **trained on the full corpus**
(`cmix_134M_hybrid_32B_sparse`, step 122070) on documents drawn from inside versus beyond the
subset's span. The model saw both regions in training, so any gap is a property of the DATA.

**Mean token NLL in NATS, LOWER = easier text.**

| web shard | inside subset | beyond subset | Δ (inside − beyond) |
|---|---|---|---|
| `web-nemotron-cc-hq-p2_0` | 2.8702 | 3.3213 | **−0.4511** |
| `web-nemotron-cc-hq-p2_1` | 2.7031 | 3.1836 | **−0.4805** |

**The subset's span is ~0.47 nats EASIER than the rest of the corpus**, consistently across two
independent shards. The math sets are complete copies and cannot be biased.

No README, metadata or `.ndocs` file in the dataset tree documents an ordering, and the quality split
is by dataset NAME (`hq` / `med-hq` / `synthetic-hq`), not documented within-file. The measurement is
the only evidence, and it is unambiguous. (Caveat: 13–24 sequences per cell. The effect is 20x the
between-shard spread, so the sign and rough size are safe; the exact magnitude is not.)

### 23.3 It fully explains a 0.32-nat anomaly that looked like a model result

The 8B knob arms (subset) end at `train-lm_loss` ~**2.33**; the 32B cmix arms (full corpus) end at
**2.6544** — same model, same mixture weights, both schedules fully decayed, same tokens/step
(262,144). A 134M model cannot get BETTER loss from 4x FEWER tokens.

Predicted from the bias: `0.70` web share x `0.47` nats x `~0.85` of a full-corpus run's web tokens
lying beyond the prefix ≈ **0.28 nats**. Observed: **0.32 nats**. The prefix bias accounts for
essentially all of it.

**Consequences, in order of how badly they bite:**

1. **NO loss or Avg11 from a subset-trained arm may be compared with a full-corpus arm.** In
   particular the knob sweep's 2.33 is NOT "almost as low as the 400M Switch" (2.4095) — that
   comparison is void. Comparisons WITHIN the knob sweep are fine; all six share the subset.
2. **Any 32B/128B arm run on the subset will be incomparable to the entire existing 32B table.**
   This was the copy's whole purpose, so it is the finding that matters most.
3. `abl_P` and the staged 1B `bharat` pair point at the subset. `abl_P` was at ~44k/122070 steps when
   this was found.

### 23.4 The remedy needs NO copying — the full corpus survives

`/proj/datasets/ndehmamy-dataset-rescue/` holds the FULL `.bin` for all four datasets **as hard
links** (`links=2`, same inode as the owners' path: verified `inode=1346304513` for p2_0 and
`1350501890` for p2_1, 1,074,869,046,028 and 1,071,451,022,752 bytes). Hard links keep the inode
alive after the owners unlink their name, so deletion by them does not destroy the data. The matching
FULL `.idx` (7.60 GB / 7.33 GB) is in that same directory, correctly paired.

**So a run that needs to be comparable to the 32B table should use
`/proj/datasets/ndehmamy-dataset-rescue/<name>` as its `data_path` prefix**, with a FRESH
`data_cache_path` (the Megatron blend index is keyed on the mix and would otherwise collide with the
subset's cache).

Do NOT pair indices across directories. `full-corpus-indices/*.idx` + subset `.bin` reads past the
end of file; subset `.idx` + full `.bin` silently trains on the first 18.6% only. Neither necessarily
errors.

One trap found while acting on this: a full-corpus directory CANNOT be assembled under
`/proj/dmfexp/` by hard-linking, because `/proj/datasets` and `/proj/dmfexp` are different
filesystems and `ln` fails across them — it fails per-file and can leave a half-built directory
holding indices with no data, which is exactly the mispairing above. Use the rescue directory in
place rather than rebuilding it elsewhere.

### 23.5 Prospective confirmation, and the two arms repointed to the full corpus

§23.2's bias was measured retrospectively (scoring a trained checkpoint on both regions). The restart
supplies a PROSPECTIVE test: the SAME model and config, differing only in corpus.

**`train-lm_loss` in NATS, lower better. Same arm, same step, same schedule position.**

| arm | corpus | step | train-lm_loss |
|---|---|---|---|
| `abl_P_134M_pure_223_isoall` | subset (18.6% web prefix) | 2000 | 3.8158 |
| `abl_P_134M_pure_223_isoall_full` | FULL corpus | 2010 | **4.1081** |

**Δ = +0.2923 nats going to the full corpus**, against the **0.28 nats** predicted in §23.3 from the
prefix bias (0.70 web share x 0.47 nats x 0.85). The prediction was made before this run existed, so
this is a genuine out-of-sample confirmation. The subset is easier, by the amount measured.

**Repointed / launched 2026-09-22 23:4xZ:**

* `abl_P_134M_pure_223_isoall_full` (job 1868806, 8 GPU grp_ebm) — the `[2,2,3]` h1088 iso-everything
  pure arm, datamix now byte-identical to the 32B table (so it reuses `/proj/dmfexp/nima/.cache/megatron_cmix`,
  no index rebuild), `repulsion_coef: 0.0`, NEW `save_path` so the subset-trained weights cannot be
  resumed, and the rescued tokenizer (sha `82507a72ac9a10f0`, byte-identical to the owners' copy,
  which may be deleted). Verified: `DeviceMesh(..., 'cuda')` not cpu, 262,086 tok/step empirically,
  lr 2.0e-3 after warmup, 0.1729 s/step → **5.86 h** for 32.00B. The old subset job 1865857 was
  killed and confirmed EXIT.
* `abl_U_134M_sandwich_rnorm_none` (job 1868962, 8 GPU grp_ebm) — the sandwich with
  `routing_norm: none`, a SINGLE-variable diff from `cmix_134M_sandwich_32B_sparse`. 32.00B, ~6.3 h.

**Why the sandwich and not the hybrid:** `abl_R_134M_hyb_rnorm_none` IS the hybrid with
`routing_norm: none` — a one-line diff, already run to 32B on the full corpus — and it already wins
(oracle 44.92 vs 44.82; sparse **45.43 vs 44.67**). Rerunning the hybrid would reproduce an existing
result. The sandwich with that setting is untested and is the strongest 134M under sparse eval
(44.34), so it is the arm with something to learn.

**Why no 400M rerun:** measured 61,035-step wall times are **37.3 h** (sandwich, 2.2017 s/step) and
**140.7 h** (hybrid, 8.2999 s/step — the placement artifact of §PLACEMENT). Both exceed the time to
the deadline, so the 134M tier is where GPUs should go.

### 23.6 The knob sweep INVERTS on Avg11 — no knob has a defensible Avg11 gain

The five completed knob arms, evaluated at 8B (on the subset, so comparable only to each other).
**Avg11 and MMLU in PERCENTAGE POINTS, higher better; ppl wikitext word-level, LOWER better;
`train-lm_loss` in nats, LOWER better.**

| knob | Avg11 | MMLU | ppl | loss@8B |
|---|---|---|---|---|
| `repulsion_coef: 1.0` | **44.07** | 25.00 | 52.99 | 2.3926 |
| `routing_norm: none` | 44.05 | 24.64 | 51.66 | 2.3692 |
| `proxy_loss_coef: 0.1` | 43.69 | 24.17 | 56.57 | 2.4273 |
| `repulsion_coef: 0.0` | **43.07** | 26.26 | **50.78** | **2.3590** |

**`repulsion_coef: 0.0` is BEST on loss and perplexity and WORST on Avg11.** So §22's "repulsion off
is the sweep's one robust win" is a LOSS-ONLY claim and must be stated as such — it does not carry to
Avg11. This is the same inversion §18 documented for `routing_norm`, and it is exactly why the 8B
decision rule in `rank_arms_at_milestone.py` is validated for LOSS RANKING ONLY.

Neither metric settles it, and the whole Avg11 spread is 1.0pp on single seeds against a 0.73pp seed
range. **Conclusion: no knob in this sweep has a defensible Avg11 improvement.** The sweep's value is
negative knowledge — do not raise the auxiliary losses, and repulsion's setting barely matters because
Bug B held it at an effective 1e-4 throughout the published work.

Consequently `abl_U` was given ONLY the `routing_norm` change, not `repulsion_coef: 0.0` as well: a
clean one-variable test is worth more than stacking an unproven setting, and `repulsion_coef: 0.1`
with the default `router_aux_loss_coef` reproduces the published arms' effective ≈-off condition
anyway.

### 23.7 The watchdog RESURRECTED the killed subset arm within minutes

After `abl_P_134M_pure_223_isoall` (subset) was killed and replaced by the full-corpus arm, the
watchdog **resubmitted the subset arm** from its `watchdog_jobs.conf` line — job 1868891, 8 GPUs on
`p3-r12-n4`, training on data that cannot be compared with the 32B table. It was found only because
`bjobs` was read while checking something else.

**Killing a watchdog-managed arm is not enough; its conf line must be repointed or removed in the
same action.** `watchdog_jobs.conf` line 230 was the culprit; it now points at
`abl_P_134M_pure_223_isoall_full.yml`.

**And the dedup key is FIELD 1, which must equal the LSF job name.** `watchdog_loop.sh:353` does
`bjobs -J "$name"` with `$name` = conf field 1. The arms had been hand-submitted as `abl_P_full` and
`abl_U_sandwich_rnorm` while the conf said `abl_P_134M_pure_223_isoall_full` and
`abl_U_134M_sandwich_rnorm_none` — a mismatch that would have made the watchdog blind to both live
jobs and submit DUPLICATES on its next cycle, 16 GPUs wasted. Field 1 is now aligned with the live
job names. **Whenever an arm is registered in the conf, field 1 must be exactly the name passed to
`submit_train.sh`.**

Related: `knob_ctrl_aux1` is NOT in the conf, so its wedge required a hand resubmit (job 1869085,
4 GPUs, resumes from `global_step25000` to finish 26890 -> 30518).

The wedge itself: job 1865858 on `p6-r26-n4`, STAT=RUN, stderr 34 min stale, step frozen at 26890,
CPU time FLAT over a 30 s sample (4 busy GPUs would add ~120 s), a single `Setting OMP_NUM_THREADS`
banner so not a healthy requeue. Plain `bkill` was ignored; `bkill -r` moved it to ZOMBI for ~2 min
and then EXIT. The host reports `ok` with 0 jobs afterwards, so it was NOT promoted to
`SUSPECT_HOSTS` -- one fault on an otherwise healthy host.

### 23.8 Quantified: at 8B, loss ranking and Avg11 ranking are UNCORRELATED across knobs

The five completed knob arms, all at 8.00B on the subset corpus (comparable to each other only).
**Avg11 and MMLU are accuracy in PERCENTAGE POINTS, higher better. ppl is wikitext word-level
perplexity, LOWER better. loss is `train-lm_loss` in nats at step 30400, LOWER better.**

| knob | Avg11 | MMLU | ppl | loss@8B | loss rank | Avg11 rank |
|---|---|---|---|---|---|---|
| `repulsion_coef: 1.0` | **44.07** | 25.00 | 52.99 | 2.3926 | 4 | **1** |
| `routing_norm: none` | 44.05 | 24.64 | 51.66 | 2.3692 | 2 | 2 |
| `temperature: 0.35` | 43.94 | 25.12 | 51.84 | 2.3700 | 3 | 3 |
| `proxy_loss_coef: 0.1` | 43.69 | 24.17 | 56.57 | 2.4273 | 5 | 4 |
| `repulsion_coef: 0.0` | **43.07** | **26.26** | **50.78** | **2.3590** | **1** | **5** |

**Spearman rho(loss rank, Avg11 rank) = −0.30** (sum d² = 26, n = 5). The two metrics are not merely
noisy against each other — the ordering is slightly INVERTED. The best-loss arm is the worst-Avg11
arm and vice versa; only the two middle arms agree.

Note that **ppl tracks loss, not Avg11** (`rep_off` has both the best loss and the best perplexity).
So the divergence is specifically likelihood-vs-task-accuracy, not a perplexity computation artifact.

**What this licenses and forbids:**

* The §18 rule stands and is now quantified for knobs: an 8B read on LOSS predicts 32B LOSS ranking
  (rho = 1.000 over 7 arms at 134M), and says **nothing** about Avg11 ranking (rho = −0.30 here).
* **No knob in this sweep is actionable.** The whole Avg11 spread is 1.00pp on single seeds against a
  0.73pp seed range, so even the Avg11 ordering is weakly resolved. Claiming any knob as an
  improvement from this sweep would be reading noise.
* §22's "repulsion off is the sweep's one robust win" must be quoted as a **loss-and-perplexity**
  result only. On Avg11 it is the worst of the five.

The control (`knob_ctrl_aux1`, shipped knobs at `router_aux_loss_coef: 1.0`) wedged at step 26890 and
was resumed; its Avg11 is the missing cell that would say whether ANY of these beats the shipped
configuration at all. Until it lands, the correct statement is that no knob has been shown to beat
the shipped settings.

### 21.9 The pure arm's sparse collapse DECOMPOSES: ~43% missing mu, the rest recurrence depth

A controlled 2x2 on `cmix_134M_pure_32B_sparse` (134M, `1x12E`, 32B). The two exports differ in
EXACTLY one config field, `sinkhorn_persist_mu` (`false` in `unsharded_step122070`, `true` in
`unsharded_mucal`), so this isolates the mu hypothesis from the proxy one.

**Avg11 = 11-task accuracy mean in PERCENTAGE POINTS, higher better. ppl = wikitext word-level
perplexity, LOWER better. "sparse penalty" is the within-export sparse-minus-oracle difference.**

| export | routing | Avg11 | ppl | sparse penalty |
|---|---|---|---|---|
| mu MISSING (`persist_mu: false`) | oracle | 42.00 | 66.08 | — |
| mu MISSING | **sparse** | **34.04** | **446.72** | **−7.96pp, 6.76x ppl** |
| mu present (`persist_mu: true`) | oracle | 40.98 | 67.36 | — |
| mu present | **sparse** | **36.47** | **156.49** | **−4.51pp, 2.32x ppl** |

**Restoring mu recovers 3.45pp of the 7.96pp collapse (43%)** and cuts the perplexity blowup from
6.76x to 2.32x. **A −4.51pp penalty survives.** So:

1. **`sinkhorn_persist_mu: false` is a real and substantial defect at high iteration count**, as
   CLAUDE.md's "+1.827 nats at 12 iterations" warning predicted, and it is amplified by sparse
   routing because an untilted router changes WHICH experts are selected, not merely their weights.
2. **It is not the whole story.** The residual −4.51pp at 12 applications, against −0.15pp for the
   6-application hybrid and 0.00pp for the 4-application 400M sandwich, is consistent with a single
   proxy head (`proxy_iters: 1`) being unable to serve twelve different activation distributions —
   the failure mode `calibrate_proxy_router_20260916.py`'s docstring names. Measured support: the
   stored proxy's top-2 agreement on this arm falls from **0.71 at the first application to ~0.49**
   at later ones.
3. **Note mu recalibration slightly HURTS the oracle path** (42.00 -> 40.98), so it is not a free
   win; it trades dense accuracy for sparse robustness. Quote the within-export penalties, not
   cross-export mixtures like "mu-present sparse vs mu-missing oracle" (which would read −5.53pp).

**For the paper:** pure recurrence at depth 12 can now be described with a measured decomposition
instead of an unexplained collapse, and the actionable statement is that a recurrent arm needs
`sinkhorn_persist_mu: true` AND, above ~6 applications, more than one proxy head
(`proxy_iters` > 1) before its sparse path can be trusted.

### 23.9 FINAL VERDICT on the knob sweep: the shipped configuration wins; every knob is worse

The control completed (after its wedge) and was benchmarked. All six arms, 8.00B on the subset corpus,
comparable to each other only.

**Avg11 and MMLU are accuracy in PERCENTAGE POINTS, higher better. ppl is wikitext word-level
perplexity, LOWER better. loss is `train-lm_loss` in nats at step 30400 (median over +-300), LOWER
better. dAvg11 is the arm minus the CONTROL, in pp.**

| arm | Avg11 | MMLU | ppl | loss | loss rank | Avg11 rank | dAvg11 |
|---|---|---|---|---|---|---|---|
| **CONTROL (shipped knobs)** | **44.37** | 25.19 | 51.93 | 2.3719 | 4 | **1** | — |
| `repulsion_coef: 1.0` | 44.07 | 25.00 | 52.99 | 2.3926 | 5 | 2 | −0.30 |
| `routing_norm: none` | 44.05 | 24.64 | 51.66 | 2.3692 | 2 | 3 | −0.32 |
| `temperature: 0.35` | 43.94 | 25.12 | 51.84 | 2.3700 | 3 | 4 | −0.43 |
| `proxy_loss_coef: 0.1` | 43.69 | 24.17 | 56.57 | 2.4273 | 6 | 5 | −0.68 |
| `repulsion_coef: 0.0` | 43.07 | **26.26** | **50.78** | **2.3590** | **1** | **6** | **−1.30** |

**Spearman rho(loss rank, Avg11 rank) = −0.31** (sum d² = 46, n = 6).

**NO KNOB BEATS THE SHIPPED CONFIGURATION.** The control is 1st on Avg11 and 4th on loss;
`repulsion_coef: 0.0` is 1st on loss AND 1st on perplexity AND last on Avg11, by 1.30pp.

**§22 is hereby corrected.** Its conclusion "repulsion off is the sweep's one robust win" is a
LOSS-AND-PERPLEXITY result only. On task accuracy it is the worst of the six. Do not quote §22 as
evidence for a setting change. The `repulsion_coef: 0.0` already set on `abl_P_full` is within noise
of the control (−1.30pp is the 8B-subset reading; the effective published setting was ~1e-4, i.e.
already ≈ off) and is not worth a second restart, but it must not be described as an improvement.

**Standing rule this reinforces:** `rank_arms_at_milestone.py`'s 8B rule is validated for LOSS
ranking of ARCHITECTURES (rho = 1.000 over 7 arms at 134M). For KNOBS, loss ranking is
anti-correlated with Avg11 ranking (rho = −0.31 over 6). Never select a knob on loss.

Note also that `repulsion_coef: 0.0` has the BEST MMLU (26.26) alongside the WORST Avg11, further
evidence that at this scale the aggregate metrics pull apart and single-metric knob selection is
unsafe in either direction.

### 21.10 abl_C (the TRUE sandwich) measured, and the sparse penalty tracks ROUTED FRACTION

`abl_C_134M_1G1x6E1G_isototal` had completed all 122,070 steps and been **never evaluated** — it was
blocked by my own `_svd_done_buf` strict-load regression (see CRITIC_LOG (g)). With the back-compat
hook it unsharded cleanly and both paths ran.

**Avg11 in PERCENTAGE POINTS, higher better; ppl wikitext word-level, LOWER better. 32B, full corpus.**

| routing | Avg11 | MMLU | ppl |
|---|---|---|---|
| oracle | 42.93 | 24.93 | 49.90 |
| sparse | 41.66 | — | 65.60 |

**ΔAvg11 = −1.27pp.** Two findings:

**(a) The thin-bread/thick-core sandwich does not pay off.** At 42.93 oracle the TRUE `1G1x6E1G`
sandwich is *below* the arm long mislabelled "sandwich" (`5G1x6E1G` block-position variant, 43.45) and
**1.89pp below the `6G1x6E` hybrid** (44.82). It is second-worst in the 134M tier, ahead only of pure
recurrence. Its perplexity is the group's BEST (49.90), another instance of ppl and Avg11 disagreeing.
Caveat when tabulating: `abl_C` is iso-**total**, carrying 98M active against the hybrid's 123M, so
this is not a like-for-like loss.

**(b) The sparse penalty tracks the fraction of block applications that are proxy-routed**, better
than it tracks recurrence count. `routed% = routed applications / all applications`:

| arm | routed% | max iters | ΔAvg11 |
|---|---|---|---|
| 1B stacked | 33% | 1 | −0.37 |
| 134M w1w2 surrogate | 50% | 6 | −0.08 |
| 134M hybrid | 50% | 6 | −0.15 |
| 400M hybrid | 50% | 6 | −0.30 |
| 134M hyb rnorm=none | 50% | 6 | +0.51 |
| 134M block-pos variant | 50% | 6 | +0.89 |
| 400M sandwich | 67% | 4 | +0.00 |
| **134M TRUE sandwich** | **75%** | 6 | **−1.27** |
| **134M pure rec** | **100%** | 12 | **−7.96** (−4.51 with mu) |

**At routed% <= 67% the penalty is indistinguishable from zero**: 7 arms, mean +0.07pp, range −0.37 to
+0.89, entirely inside the 0.73pp seed range. It becomes material only above that.

**The one controlled pair** holds recurrence depth fixed at 6 applications and varies routed fraction:
hybrid (50%) −0.15pp against true sandwich (75%) −1.27pp. Recurrence count alone cannot explain that.

**State this as suggestive, not established.** It is n=1 for the controlled pair; `abl_C` differs from
the hybrid in more than routed fraction (3 layers vs 7, different width/params); and the pure arm
confounds routed fraction with depth completely. The DEFENSIBLE claim is the threshold —
**sparsity is free while at most about two-thirds of block applications are routed** — which is a
design rule and covers every configuration the paper advocates. The mechanism (non-routed layers
absorbing the proxy's substitution error) is a hypothesis this pair supports.

Also note the w1w2 surrogate arm's `mlp_type` is `EnergyFF_SurrogateBoltzmannMoE`, NOT
`EnergyFF_BoltzmannMoE`, so any prefix match on the latter silently misses it — it initially reported
0% routed instead of 50%.

### 21.11 Wave C: the sparse penalty is ZERO AT EVERY BUDGET, and the "improvements" are noise

Six milestone exports re-evaluated on both routing paths (unblocked once all 16 finals were redone).
**Values are within-checkpoint sparse-minus-oracle Avg11 in PERCENTAGE POINTS.** Only the within-
checkpoint delta is meaningful: absolute Avg11 is non-monotone across milestones (hybrid 43.59 at 8B,
42.94 at 16B, 44.82 at 32B) because intermediate checkpoints sit mid-cosine at high LR.

| arm (both 50% routed) | 8B | 16B | 24B | 32B |
|---|---|---|---|---|
| 134M hybrid `6G1x6E` | −0.35 | +0.19 | — | −0.15 |
| 134M block-pos `5G1x6E1G` | +0.20 | −0.10 | −0.05 | +0.89 |

**Mean over all seven measurements: +0.09pp.** There is NO trend with training budget — the penalty is
scattered about zero from the earliest milestone on.

**Two corrections to earlier framing in this file:**

1. **§21.7's "co-adaptation accumulates" reading is not supported and is not needed.** The penalty does
   not shrink as training proceeds; it is already ~0 at 8B. This makes the claim STRONGER, not weaker:
   free sparsity does not require a long run. Whatever co-adaptation does, it is established early
   (the dense warmup is only 300 steps) rather than accumulated.
2. **The positive deltas are NOISE and must stop being described as improvements.** §21.7 said "four
   arms improve" and §21.10 leaned on the block-pos arm's +0.89. That arm's own milestone series is
   +0.20, −0.10, −0.05, **+0.89** — the final value is an outlier draw, not a gain. The correct
   statement everywhere is **indistinguishable from zero**, with a per-measurement scatter of about
   +-0.4pp at 50% routed (7 measurements, range −0.35..+0.89, sd ~0.39pp).

That scatter estimate is itself useful: it is an empirical noise floor for the orc-vs-sps delta, and it
means the −1.27pp at 75% routed (abl_C) is ~3 sd from zero while every 50%-routed reading is within
1 sd. So the routed-fraction threshold survives this correction, and is now better calibrated.

### 21.12 The collapse is STRUCTURAL, not developmental — so it is detectable at 8B

The last wave-C milestone completes the budget series. **Within-checkpoint sparse-minus-oracle Avg11 in
PERCENTAGE POINTS:**

| arm | routed% | 8B | 16B | 24B | 32B |
|---|---|---|---|---|---|
| 134M hybrid `6G1x6E` | 50% | −0.35 | +0.19 | — | −0.15 |
| 134M block-pos `5G1x6E1G` | 50% | +0.20 | −0.10 | −0.05 | +0.89 |
| **134M pure rec `1x12E`** | **100%** | **−6.90** | — | — | **−7.96** |

**The pure arm already loses 6.90pp at 8B** (oracle 39.93 -> sparse 33.03) and 7.96pp at 32B. Four times
the tokens changes the penalty by ~1pp on a ~7pp effect. Meanwhile both 50%-routed arms are flat within
noise at every budget.

**So the sparse penalty is a property of the ARCHITECTURE, not of training duration.** Neither the good
case nor the bad case moves materially between 8B and 32B.

**This is the cheapest useful consequence of the whole investigation:** a sparse configuration can be
validated at **8B**, a quarter of the budget, instead of after a full 32B run. The pure arm's defect
would have been caught for 8B of tokens had anyone evaluated the sparse path at a milestone — which
nobody could, because until 2026-09-22 the sparse path never ran at eval (§21.1).

Recommended standing practice: **for any new sparse arm, run the sparse eval at its 8B milestone and
compare with the oracle eval at the same checkpoint.** A delta beyond ~1pp (about 3 sd of the 0.37pp
noise floor, §21.11) means the configuration is broken and the remaining 24B of tokens would be wasted.
This costs one extra eval on a checkpoint the milestone machinery already preserves.

### 23.10 CORRECTION: the "0.73pp seed range" I cited all session does NOT exist

Sections 21.7-21.12 and 23.8-23.9 as originally written repeatedly compared effects against "the
0.73pp seed range". **That is not a seed range.** It is the ARM-TO-ARM gap from §15.7
("It leads the energy hybrid by +0.73pp Avg11"). §15.7's own first caveat says the opposite of what I
used it for:

> **SINGLE SEED, NO ERROR BARS.** The multi-seed arms (`iclr_*_s1234`/`_s7`) have been PAUSED in
> `watchdog_jobs.conf` since 2026-09-12, so seed spread at this scale is **UNQUANTIFIED**.

So there was no measured seed-noise threshold in this project, and I invented one by misreading a
between-arm margin as a within-arm spread. (§13 separately mentions a "0.32pp matched-seed spread";
its provenance has NOT been re-verified and it should not be quoted until it is.)

**What survives.** The paper does not cite 0.73 anywhere — checked. The threshold it actually uses is
the **0.37pp** standard deviation MEASURED in §21.11 from seven within-checkpoint oracle-vs-sparse
deltas, which is a genuine same-checkpoint dispersion and is the right yardstick for that comparison.
Every quantitative claim in the pushed paper rests on that number, not on 0.73.

**What must be restated.** Anywhere in this file reading "within the 0.73pp seed range", substitute
"within the 0.37pp measured dispersion of the orc-vs-sps delta (§21.11)" — and note that this bounds
the SPARSE-vs-ORACLE comparison on one checkpoint, NOT the seed-to-seed reproducibility of an arm's
absolute Avg11. The latter remains **unquantified at 134M**, which means:

* the knob sweep's 1.00pp Avg11 spread (§23.9) cannot be declared "inside seed noise" — the correct
  statement is that no knob beats the control and the spread is uncontrolled for seed;
* `abl_R`'s +0.10pp oracle lead over the hybrid is NOT established, though its +0.76pp sparse lead
  exceeds the 0.37pp same-checkpoint dispersion and is the better-supported half of that result;
* §21.10's routed-fraction threshold is unaffected: it is built on within-checkpoint deltas against
  the 0.37pp dispersion, not on cross-seed comparisons.

**The fix worth GPU time:** a second seed of the best arm would convert several of these from
"suggestive" to "established". That is a better use of 8 GPUs than finishing an arm whose expert
width is already known to be unviable (§23.11).

### 23.11 abl_P's iso-everything constraint is INFEASIBLE — its experts are 31-47x too narrow

`abl_P_full` ran worse than every other arm at every matched step. Both it and
`cmix_134M_pure_32B_sparse` are on the FULL corpus at 262,144 tok/step, so the comparison is valid.
**`train-lm_loss` in NATS, median over +-500 steps, LOWER better:**

| step | `abl_P_full` `[2,2,3]` h1088 | `pure 1x12E` h768 | gap |
|---|---|---|---|
| 10000 | 3.5787 | 3.4776 | +0.101 |
| 20000 | 3.4791 | 3.3476 | +0.132 |
| 30000 | 3.4256 | 3.2907 | +0.135 |
| 40000 | 3.3955 | 3.2500 | +0.146 |
| 50000 | 3.3657 | 3.2142 | **+0.152** |

The gap WIDENS monotonically. The curve shape is healthy (monotone descent, no plateau or spike), so
this is a weaker model, not a broken run.

**Cause: per-expert width.** `intermediate_size` is the TOTAL across experts, so:

| arm | K | I_total | per-expert $I_e$ | $I_e$/hidden |
|---|---|---|---|---|
| `abl_P_full` block0 | 32 | 4608 | **144** | **0.13** |
| `abl_P_full` blocks1-2 | 32 | 3072 | **96** | **0.09** |
| `pure 1x12E` | 16 | 71680 | 4480 | 5.83 |
| `hybrid 6G1x6E` | 16 | 16384 | 1024 | 1.33 |

**31-47x narrower than the arms it is compared with.** This is the documented iso-param failure: this
file's own header says of the B-series "experts too small -- fails" and "do not scale the iso-param
design ... use full-size experts". The LR (2e-3 at hidden 1088 vs 768) may contribute a little but
cannot produce a monotonically widening 0.152-nat gap.

**The constraint is structurally infeasible.** With THREE distinct MoE blocks you cannot have all of:
iso-total 134M params, K=32, and viable expert width. Three blocks must share the parameter budget one
block gets in the hybrid, so per-expert width collapses. Restoring width requires giving one up:
K=4 gives $I_e$=1152 (1.06x hidden) but k/K=0.5, which abandons the sparsity the paper is about.

**MY PRE-FLIGHT MISS.** The five-point config check (CLAUDE.md) covers datamix, parent, structure,
budget and param TOTALS — it does not ask for per-expert width, and I audited totals (134.10M /
123.13M / 141.70M, all within 0.11%) and passed it. Totals matched precisely BECAUSE the experts were
shrunk. **Add per-expert $I_e$ and $I_e$/hidden to the pre-flight**: a config can be iso-param to
0.1% and still be architecturally dead.

### 23.12 Watchdog race: registering and hand-submitting in the same minute ALWAYS duplicates

`abl_R_s7` ended up with two live jobs, same config, same `save_path`:

```
1872007 RUN  Sep 23 02:50:11   <- watchdog, from the conf line I had just added
1872009 PEND Sep 23 02:50:18   <- my hand submission, 7 s later
```

The watchdog's dedup (`watchdog_loop.sh:353`, `bjobs -J "$name"`) ran at 02:50:10 and correctly saw no
live job, because mine did not exist yet. Two writers on one `save_path` would have written checkpoints
over each other; caught within two minutes only because §23.11's duplicate detector was armed. Mine was
still PEND so nothing had been written; killed 1872009, kept 1872007.

**Getting conf field 1 right (§23.7) prevents steady-state duplicates but NOT this race.** There are
exactly two safe orders:

1. **Submit first, confirm `bjobs -J <name>` shows it, THEN add the conf line.**
2. **Add the conf line and let the watchdog launch it — do not hand-submit at all.**

Adding the line and submitting concurrently is the one ordering that always races. Option 2 is simpler
and is what the watchdog exists for; option 1 is for when the arm must start within seconds.

**Check after any registration:** `bjobs -noheader -o "jobid stat" -J <name>` must return exactly one
row. Do it for every arm whose conf line was touched, not just the one you launched.

### 23.13 MEASURED ON REAL STEPS: sparse_forward is a net LOSS at k/K=0.50 and a net WIN at k/K=0.25

Both new arms flip from the dense fused path to the sparse path at `sparse_start_step: 300`, which gives
a within-run A/B on identical hardware, identical code and identical tokens/call (mbs 2 x 4096 = 8192).
**Median `train-step_time` in SECONDS, lower better.**

| arm | K | k | k/K | I_e | dense (steps 50-299) | sparse (steps >900) | ratio |
|---|---|---|---|---|---|---|---|
| `abl_V_223_h768` | 8 | 4 | 0.50 | 2911 | 0.233 | **0.377** | **1.61x SLOWER** |
| `abl_W_pure5x7` | 8 | 2 | 0.25 | 5248 | 0.303 | **0.222** | **0.73x, i.e. 1.37x FASTER** |

This confirms §12.9's cost model on real training steps rather than a microbenchmark: dispatch cost is
**O(T k)** while the arithmetic saved is **O(T (K-k) I_e)**. Numerically:

* `abl_V`: saving ∝ (8-4) x 2911 = 11,644, overhead ∝ 4
* `abl_W`: saving ∝ (8-2) x 5248 = 31,488, overhead ∝ 2 — **2.7x more saving at half the overhead**

So raising k to lift ACTIVE toward the hybrid (§23.11's iso-ACTIVE compromise) costs `abl_V` 61% of its
throughput for sparsity that saves little. **Design rule: the sparse path only pays off at low k/K; at
k/K >= 0.5 run dense.** This is the throughput counterpart to §21.10's accuracy rule (sparsity is free
while routed fraction <= 2/3) and the two together bound where the mechanism is worth using.

Caveat: the dense-phase medians come from steps 50-299, early enough that `torch.compile` may not be
fully settled, and the flip at step 300 forces one recompile. `abl_V`'s 400-900 median (0.399) vs >900
(0.377) shows it settling, so the ratio is stable to ~5%. The two arms also differ in `I_e` as well as
k, so this is the joint effect of the cost model's two terms, not a k-only ablation.

**No action taken:** at 0.377 s/step `abl_V` still completes 122,070 steps in ~12.5 h (~15:55Z), about
8 h inside the deadline, and restarting it would forfeit the sparse-eval datapoint.

### 23.14 RETRACTION of half of §23.13 — abl_V's slowdown is HOST CONTENTION, not sparsity

§23.13 reported that `sparse_forward` costs 1.61x throughput at k/K=0.50, from `abl_V`'s dense-vs-sparse
phases. **That number is withdrawn.** `abl_V` shares host `p5-r07-n2` with two jobs from another user
(`zwhong`: `ocRSQ2-1` on 10 slots, `ocWM2-6` on 8), and its step time OSCILLATES rather than degrades:

| step window | median s/step |
|---|---|
| 50-300 | 0.233 |
| 300-600 | 0.587 |
| 600-900 | 0.361 |
| 900-1200 | 0.388 |
| 1200-1500 | 0.590 |
| 1500-1800 | 0.404 |

Alternating fast and slow windows on the same configuration is the signature of the documented
**placement artifact** (`PLACEMENT_ARTIFACT_20260915`: 2.7x from placement alone, dataloader starvation
from shared CPU slots), not of the sparse path. The dense phase happened to fall in a quiet window and
the sparse phase in a contended one, so the 1.61x is contention, not mechanism. This violated the
project's own standing rule — "no s/step claim is admissible unless placement-controlled" — recorded in
CLAUDE.md, four minutes after I wrote the claim down.

**Co-tenancy audit of all four running arms** (foreign = jobs from other users on the same host):

| arm | host | foreign jobs | median s/step | p10 | p90 | spread |
|---|---|---|---|---|---|---|
| `abl_V_223_h768` | p5-r07-n2 | **2** | oscillating 0.23-0.59 | — | — | **~2.5x** |
| `abl_W_pure5x7` | p2-r15-n4 | **0** | 0.222 | 0.221 | 0.298 | 1.35x |
| `abl_R_s7` | p2-r30-n1 | 0 | 0.149 | 0.149 | 0.150 | **1.01x** |
| `abl_U_sandwich_rnorm` | p2-r01-n2 | 4 | 0.150 | 0.150 | 0.151 | **1.01x** |

**What survives:** `abl_W`'s dense 0.303 -> sparse 0.222 = **1.37x FASTER at k/K=0.25**, measured on a
host with ZERO foreign jobs and a 1.01-1.35x internal spread (the 1.35 is its own dense->sparse
transition). That half is placement-clean and is the first demonstration in this project of
`sparse_forward` winning on real training steps rather than in a microbenchmark. The k/K=0.50 side needs
a placement-controlled rerun before anything is claimed about it.

**Also worth keeping:** `abl_U` has FOUR foreign jobs and a 1.01x spread, so co-tenancy is not
automatically harmful — it depends on what the neighbours do. Counting co-tenants is a screen, not a
verdict; the step-time spread is the actual measurement.

**abl_V's schedule:** its quiet-window rate is ~0.38 s/step, so ~12.7 h remaining (~16:15Z), inside the
deadline. Left running; if contention worsens it is the arm at risk, and it is also the least valuable
of the five (abl_W tests the same sparsity thesis on a clean host).

### 24.1 `routing_norm: none` ELIMINATES the block-placement penalty (abl_U, 2026-09-23)

`abl_U_134M_sandwich_rnorm_none` = the block-position variant `5G1x6E1G` with `routing_norm: none`,
a single-variable diff. **Avg11 in PERCENTAGE POINTS, higher better; ppl wikitext word-level, LOWER
better; 32B tokens, full corpus, ORACLE routing.**

| arm | zscore | none | gain |
|---|---|---|---|
| hybrid `6G1x6E` | 44.82 | 44.92 (`abl_R`) | +0.10 |
| block-pos `5G1x6E1G` | 43.45 | **44.94** (`abl_U`) | **+1.49** |
| **placement gap** | **1.37pp** | **0.02pp** | — |

`abl_U`: Avg11 44.94, MMLU 25.63, ppl 41.99, final `train-lm_loss` 2.7012.

**The 1.37pp block-placement penalty of §14.1 vanishes under `routing_norm: none`** — the two
architectures become indistinguishable (44.94 vs 44.92, 0.02pp). So:

1. **§14.1's "block PLACEMENT costs 1.37pp, more than the routing mechanism's 0.05pp" is
   zscore-SPECIFIC** and must be qualified wherever it is quoted. Placement is not intrinsically
   expensive; zscore normalisation makes the model placement-sensitive.
2. **`routing_norm: none` is not the marginal +0.10pp knob the hybrid alone suggested.** Judged on
   the hybrid it looks like noise; judged on the arm that zscore was hurting it is +1.49pp. It is
   also ~24% faster (0.148 vs 0.195 s/step).
3. **Third independent instance of the loss/Avg11 inversion for this knob**: `abl_U`'s loss is WORSE
   than its parent's (2.7012 vs 2.6544, +0.047 nats) while its Avg11 is 1.49pp BETTER. The other two
   are the 8B knob sweep (§23.9) and `abl_R` at 32B. Any loss-based selection would have rejected
   the best setting three times.

**Caveats:** single seed per cell — `abl_R_s7` (seed 7, ~59% done) gives the first replicate, and
seed spread at 134M remains unquantified (§23.10). The sparse-path eval for `abl_U` is still running;
under sparse routing `abl_R` gained further (44.92 -> 45.43), so `abl_U`'s sparse number is worth
waiting for before writing this into the paper.

### 24.2 `routing_norm: none` does NOT collapse routing — zscore was the thing causing imbalance

Checked because removing a normaliser is the obvious way to get expert collapse. It does the opposite.
K=16, uniform share = 0.0625. **Median of the last 200 logged samples per arm.**

| arm | effK /16 | max_share | **min_share** | entropy | n_dom |
|---|---|---|---|---|---|
| hybrid (zscore) | 13.79 | 0.1029 | **0.0026** | 0.4355 | -- |
| block-pos (zscore) | 15.58 | 0.0944 | 0.0440 | 0.4856 | -- |
| `abl_R` hyb rnorm=NONE | **15.83** | **0.0840** | 0.0513 | **0.5894** | 16.0 |
| `abl_U` blockpos rnorm=NONE | 15.54 | 0.1207 | 0.0524 | 0.5876 | 16.0 |
| `abl_R_s7` seed 7, rnorm=NONE | 15.66 | 0.1065 | 0.0530 | 0.5881 | 16.0 |

**The decisive column is `min_share`.** The zscore hybrid starves one expert to **0.26%** of tokens;
every rnorm=none arm holds its least-used expert at **5.1-5.3%**, a ~20x improvement, with
`n_dominant_experts = 16.0` (all sixteen win the argmax somewhere). Entropy RISES (0.588 vs 0.436),
i.e. routing gets SOFTER, not sharper. `effK` 15.54-15.83 of 16 is near-uniform.

So zscore normalisation was itself producing the imbalance, which is consistent with it also producing
the placement sensitivity (§24.1): both are symptoms of a router whose scale is being distorted.
The seed-7 replicate reproduces it (15.66 / 0.0530), so this is not a one-seed artifact.

**This removes the obvious objection to §24.1.** `routing_norm: none` is better on Avg11 (+1.49pp on the
arm zscore was hurting), better balanced, and ~24% faster. The one thing it is worse on is loss --
which for this knob has been the wrong signal three times.

### 24.3 gen_table1 dropped a row because the LSF job name != the save_path basename

`tokens_per_step()` globs `~/bsub_logs/<save_path basename>_*.stderr`. `abl_U`'s save_path basename is
`abl_U_134M_sandwich_rnorm_none` but it was SUBMITTED as `abl_U_sandwich_rnorm`, so no log matched and
the row was omitted with "cannot determine tokens/step -- omitted rather than guessed". The refusal was
correct behaviour; the lookup was too narrow.

Fixed by preferring the **launch ledger**, which `submit_train.sh` keys on the CONFIG PATH and which
already records `gpus` and `tokens_per_step` at submission (abl_U: 8 GPUs, 262144, 31999918080). That is
immune to the naming mismatch. **Lesson: pass `submit_train.sh` a job name equal to the config/save_path
basename, or the ledger is the only link between an arm and its budget.**

### 24.4 Table 1's pure row is NOT iso-compute — replace it with abl_W if abl_W wins (user, 2026-09-23)

Table 1 currently uses `cmix_134M_pure_32B_sparse` (`1x12E`) as the pure-energy row. **That arm is not
compute-matched to the hybrid it is compared against:**

| arm | ACTIVE | FLOPwt | apps/blocks | TOTAL |
|---|---|---|---|---|
| hybrid `6G1x6E` | 123.2M | **141.7M** | 1.400 | 134.3M |
| `1x12E` pure (current row) | 86.1M | **185.1M** | 12.0 | 134.2M |
| `abl_W` pure deep 5x7 | 127.2M (+3%) | **147.2M (+4%)** | **1.400** | 248.1M (+85%) |

`1x12E` spends **31% MORE FLOPwt per token** than the hybrid and still places last in the tier, so the
row flatters the hybrid: it is a worse arm that also had more compute. `abl_W` was built precisely to
remove that confound -- 5 distinct energy blocks x `[1,1,1,2,2]` = 7 applications gives
`apps/blocks = 1.400`, matching the hybrid's non-embedding FLOPwt/ACTIVE ratio exactly.

**Action when `abl_W` completes** (~50% as of 07:4xZ, ~4 h remaining): evaluate both paths, and if it
beats `1x12E` make it Table 1's pure row.

**Two things the caption must then say, or the swap becomes its own distortion:**
1. `abl_W` is iso-**COMPUTE**, not iso-total: 248.1M total against 134.3M (+85%). A sparse stack cannot
   match total, active and FLOPs simultaneously (§23.11 -- the search bottoms out at 20.2% because the
   hybrid's non-embedding active fraction is 0.807, high because six of its seven blocks are dense).
   So the row would mean "pure energy at the hybrid's cost", not "at the hybrid's size".
2. `1x12E` stays in Table 6 regardless. Its sparse collapse (34.04 Avg11, ppl 446.72 against 42.00 /
   66.08 oracle) is a finding in its own right, and ~43% of it is an unpersisted Sinkhorn potential
   rather than the architecture (§21.9).

**If `abl_W` is WORSE, leave the row and say so.** A fairly-provisioned pure energy stack losing is a
STRONGER negative result than an over-provisioned one losing, and it should be reported as such rather
than quietly kept out of the table.

### 25.1 SEED SPREAD AT 134M, MEASURED AT LAST — and it breaks the Switch-parity claim

`abl_R_s7` (seed 7, single-variable from `abl_R`) completed 32B. **Avg11 in PERCENTAGE POINTS, higher
better; ppl wikitext word-level, LOWER better.**

| arm | oracle | ppl | **sparse (deployed)** | ppl |
|---|---|---|---|---|
| `abl_R` seed 42 | 44.92 | 42.36 | **45.43** | 42.79 |
| `abl_R_s7` seed 7 | 45.08 | 41.60 | **44.56** | 42.82 |
| **spread** | **0.16pp** | 0.76 | **0.87pp** | 0.03 |

**This is the first seed-spread measurement at 134M** -- §23.10 recorded that it was UNQUANTIFIED and
that the 0.73pp I had been citing was a misread arm-to-arm gap. Now measured: **0.16pp on the oracle
path, 0.87pp on the sparse path.**

**Consequence 1: the Switch-parity claim is withdrawn.** I pushed Table 1 with `abl_R` = 45.43 and
described it as exact parity with the FLOP-matched Switch (45.43). That was the HIGHER of two seeds.
Two-seed mean is **45.00**, so Switch leads by **~0.43pp**. Report "close to, but behind, the learned
gate", never parity.

**Consequence 2: the sparse path is ~5x noisier across seeds than the oracle path** (0.87 vs 0.16pp).
That is mechanically sensible -- expert selection is discrete, so a seed change can flip which experts
a token receives, whereas the oracle path averages over all K. It also means:
* the 0.37pp WITHIN-CHECKPOINT dispersion of §21.11 is NOT a seed bound and must not be used as one;
* **every single-seed sparse comparison in this project carries ~0.9pp of seed noise**, which swallows
  the +0.51pp (`abl_R`), +0.89pp (block-pos) and +0.18pp (`abl_U`) sparse "gains" reported earlier.
  The routed-fraction threshold (§21.10) survives because its effects are 1.27pp and 7.96pp, i.e. 1.5x
  and 9x the seed spread, but the small positive deltas do not.

### 25.2 All THREE 100%-routed arms collapse under sparse inference

| arm | routed | oracle | sparse | sparse ppl |
|---|---|---|---|---|
| `cmix_134M_pure_32B_sparse` `1x12E` | 100% | 42.00 | 34.04 | 446.72 |
| `abl_W_pure5x7` deep, iso-compute | 100% | 40.02 | **35.91** | **306.11** |
| `abl_V_223_h768` iso-total | 100% | 41.38 | **34.37** | **996.98** |

Three independently-designed pure-energy arms -- one recurrent x12, one 5 distinct blocks x 7
applications matched on compute, one `[2,2,3]` matched on total -- all collapse. The failure is
therefore a property of routing the WHOLE network through the cheap selector, not of any one arm's
provisioning, which is exactly what §21.10's routed-fraction rule predicts.

**For the Table 1 pure row (§24.4):** `abl_W` IS better on the deployed path (35.91 vs 34.04) and is
the fairer comparison, so it qualifies for the swap. But the honest framing is that iso-compute does
not rescue pure energy -- both numbers are unusable, and swapping trades one broken figure for a
slightly less broken one. Recommend reporting `abl_W` with that stated, not silently.

---

## 26. Full corpus copied to a shared, independent, validated location (2026-09-23)

### 26.1 What exists now

`/proj/dmfexp/datasets-shared/granite-4-cmix-FULL/` — a **real copy** (not hard links) of all four
cmix datasets, on the `dmfexp` fileset, i.e. a different fileset from the owners' data and from
`/proj/datasets/ndehmamy-dataset-rescue`. 2.28 TB. This is now the path every new run should use.

| dataset | bytes (.bin) | tokens | docs |
|---|---|---|---|
| `web-nemotron-cc-hq-p2_0` | 1,074,869,046,028 | 268,717,261,507 | 379,955,333 |
| `web-nemotron-cc-hq-p2_1` | 1,071,451,022,752 | 267,862,755,688 | 366,542,481 |
| `megamath-web-pro_0` | 52,045,089,268 | 13,011,272,317 | 14,978,856 |
| `finemath-3plus-rewritten_0` | 84,474,794,208 | 21,118,698,552 | 21,405,610 |
| **total** | **2,282,839,952,256** | **570,709,988,064** | **782,882,280** |

Columns: raw bytes; token count read from the `.idx`; document count. bytes/token = 4.00 (int32) for
all four. Token counts match the source `.idx` EXACTLY.

Also present: matching `.idx` and `.ndocs` for each, and `tokenizers/granite-4.0-tiktoken`.

### 26.2 Why a size match was not enough, and what was actually verified

The failure this had to exclude is a `.bin`/`.idx` pair that is individually well-formed and mutually
wrong. Full `.idx` + subset `.bin` reads past EOF; subset `.idx` + full `.bin` silently trains on the
first 18.6%. **Neither necessarily errors.** So `scripts/validate_full_corpus_20260923.sh` checks
five things, all of which PASSED:

1. byte sizes of every `.bin`/`.idx`/`.ndocs` against the source — 12/12 match;
2. no leftover `.part` file (an interrupted copy);
3. `cmp` on the first, middle and last 8 MiB of every `.bin`;
4. **the token count the `.idx` reports, against the source `.idx`, plus bytes/token and an actual
   decode of the FIRST and LAST document** — this is the decisive one, and it is what a size check
   cannot do;
5. tokenizer present.

Then an end-to-end **40-step 2-GPU training smoke test** (job 1884982, `configs/cmix/_smoke_shared_FULL.yml`):
GPU `DeviceMesh(ddp=2)`, 0 CUDA errors, loss 7.936 -> 7.681 over steps 20-40, and the trainer's own
`Tokens per epoch` lines report the FULL shards (`p2_1` 265,485,990,821 = 99% train split of 267.9B).
**A subset-backed run would report ~50B there** — so that line is the cheapest possible check that a
config is on the full corpus, and it is worth reading on every new launch.

### 26.3 Protection, and the honest limits of it

- Files are **mode 444**, directory **not group-writable** — so nobody, including us, can truncate
  them in place. This is the failure mode a hard link does NOT defend against.
- Different fileset from the owners' data, so a purge of `datasets` does not take it.
- Readable by POSIX group `proj_dmfexp`: bsaha3, mau, bharat, csabath, ndehmamy — verified.
  **rpanda is NOT in `proj_dmfexp` and cannot read it.** Note LSF `grp_ebm` is an LSF scheduling
  group and governs nothing about file access; the two were conflated earlier.
- Remaining exposure: it is one copy on one filesystem. There is no second copy.

### 26.4 The copy was slow because of our own script, not the filesystem

The first attempt averaged **106 MiB/s** and would have taken ~6 h. Cause: the script wrapped `cp` in
`ionice -c2 -n7` — the LOWEST I/O priority — and ran a single stream with the default block size.

`scripts/copy_corpus_fast_20260923.sh`: one `dd bs=64M` stream per file, in parallel, **no `ionice`**.
Measured **~7.2 GB/s aggregate, 3.2 GB/s single-stream**; the whole 2.28 TB completed in **5 m 16 s**,
and the last 11 GiB of the already-started shard took **4 seconds**. That is a ~68x difference from
one wrapper. Writes to `.part` and renames only on an exact byte-size match, so it is resumable, and
it resumed the interrupted 989.6 GiB shard from its offset rather than recopying it.

Two mistakes worth not repeating: the first version of the fast script passed dataset names without
the `.bin` extension, so every `stat` failed and no stream started (caught in 30 s because the log
showed `No such file or directory`); and `pkill -f copy_full_corpus_20260923.sh` matched its own
command line and killed the calling shell. Use a PID, or a pattern that cannot match the killer.

### 26.5 Config status

- `configs/RECOMMENDED_400M_hybrid_best.yml` — points at the shared FULL copy, the shared blend cache
  and the shared tokenizer. Header carries the evidence for `routing_norm: none` and `mbs 4 / ga 4`,
  and the "LAUNCH AT EXACTLY 8 GPUs" warning. `save_path` is `/CHANGE/ME/...` on purpose.
- `configs/cmix/_smoke_shared_FULL.yml` — the 40-step smoke config, for re-validating after any move.
- Shared blend cache: `/proj/dmfexp/datasets-shared/.cache_megatron_cmix_FULL` (mode 2775).
- The live arms (`abl_X`, `abl_Y`, `abl_Z1`, `abl_Z2`) still read
  `/proj/datasets/ndehmamy-dataset-rescue`. **Do not repoint a running arm** — the blend index is
  keyed on the paths, so changing them mid-run rebuilds the index and changes the sample order. They
  are reading the same bytes either way.

### 26.6 Arm status at the time of writing (2026-09-23 ~18:55Z)

Columns: median s/step over the last 20 logged steps; current step / total; hours of wall-clock left
at that rate. tokens/step verified EMPIRICALLY from the log
(`billion_tokens_per_day * 1e9 * step_time / 86400`), not read off the config.

| arm | job | s/step | step | tokens/step | ETA |
|---|---|---|---|---|---|
| `abl_Z1_allmoe_hyb` (6S1x6E) | 1884853 | 0.357 | 290/122070 | 262,146 | 12.1 h |
| `abl_Z2_allmoe_bpos` (5S1x6E1S) | 1884856 | 0.233 | 430/122070 | 262,135 | 7.9 h |
| `abl_Y_400M_sandwich` | 1876409 | 1.556 | 17050/61035 | 524,289 | 19.0 h |
| `abl_X_400M_hybrid` | 1876408 | 3.368 | 11310/61035 | 524,288 | 46.5 h |

All four: GPU `DeviceMesh(ddp=8)`, 0 CUDA errors, stderr 0 min stale. `grp_ebm` is **32/32, zero
free** — all 32 GPUs are these four arms.

**A step-time mean over a window that includes step 1 is meaningless.** `abl_Z1`'s first step took
**36.4 s** (compile); including it put the mean at 2.292 s/step against a true 0.357, which read as
"6.3x slower than Z2, will miss the deadline" and nearly triggered a kill-and-resubmit for placement.
Use a median over recent steps, and exclude the compile step.

Early routing health on `abl_Z1` (all-MoE, `routing_norm: none`), from step 140: `effective_n_experts`
15.21 of 16, `max_share` 0.112, `min_share` 0.038, `n_used_experts` 16/16. No collapse — consistent
with the 134M `routing_norm: none` finding.

---

## 27. Priority reshuffle 2026-09-23: iso-all sandwich, 6G6E rnorm=none, all-MoE deep

### 27.1 Why abl_X is slow: the model, not placement

`abl_X` (400M hybrid) runs at 3.368 s/step and `abl_Y` (400M sandwich) at 1.556 s/step, a 2.16x gap
that looked like a placement artifact. It is not. `abl_X` has `layer_iterations: [1,1,1,1,1,1,6]` =
**12 block applications**; `abl_Y` has `[1,4,1]` = **6**. The hybrid does twice the work per step.
Both arms are sparse on identical settings (`sparse_forward: true`, `sparse_start_step: 300`,
top_k 2 of K=32, `proxy_rank: 16`, mbs 4 / ga 4 so tokens/call = 16,384, the winning column), so
sparsity is not the difference either.

The same fact is the active/FLOP deficit: **`abl_Y` is iso-TOTAL with `abl_X` and 28.7% short on
ACTIVE, 27.5% short on FLOPwt.** Any "sandwich loses to hybrid" read from that pair is confounded
with a 28% compute deficit.

### 27.2 The sandwich can be matched on all three axes, and the fix is WIDER BREAD, not wider experts

Audited with `energy_ff_paramcount.audit_config` (never by hand). Parameters in millions; the goal is
EQUALITY with the hybrid, not a maximum.

| arm | TOTAL | ACTIVE | FLOPwt | apps | note |
|---|---|---|---|---|---|
| `abl_R` 134M hybrid | 134.25 | 123.24 | 141.72 | 12 | reference |
| **`abl_AA` 134M sandwich (NEW)** | **134.25** | **123.24** | **141.71** | 8 | all three to 0.00% |
| `abl_C` 134M sandwich (old) | 134.25 | 98.46 | 134.64 | 8 | iso-total only, ACTIVE -20.1% |
| `abl_X` 400M hybrid | 399.78 | 219.43 | 299.38 | 12 | reference |
| **`abl_AD` 400M sandwich (NEW, not launched)** | **399.51** | **219.16** | **299.11** | 8 | all three to 0.13% |
| `abl_Y` 400M sandwich (running) | 400.33 | 156.54 | 217.20 | 6 | iso-total only, ACTIVE -28.7% |

**The mechanism, and why the obvious fix fails.** The three axes respond differently:
`TOTAL <- K*I_e`, `ACTIVE <- k*I_e`, `FLOPwt <- R * (energy block active)`. So widening the experts
raises TOTAL `K/k` times faster than ACTIVE: 8x at 134M (K=16,k=2) and 16x at 400M (K=32,k=2).
Widening experts therefore CANNOT buy iso-active at iso-total, and that is exactly how `abl_C` and
`abl_Y` ended up iso-total only. The fix is the opposite: leave the mixture block **byte-identical**
to the hybrid's and widen the two dense bread blocks to absorb what the hybrid's other four dense
blocks carried.

- `abl_AA`: energy block identical to `abl_R` (K=16, top_k=2, `intermediate_size` 16384, I_e=1024,
  I_e/hidden 1.33, R=6). Dense bread 2048 -> 8192, I_G/hidden 2.67 -> **10.67**.
- `abl_AD`: energy block identical to `abl_X` (K=32, top_k=2, `intermediate_size` 187872, I_e=5871).
  Dense bread 4096 -> 14976, I_G/hidden 4.00 -> **14.62**. `layer_iterations` [1,4,1] -> [1,6,1] and
  `sinkhorn_mu_iters` 4 -> 6 to keep pre-flight rule 9.

Because the mixture block is unchanged, these arms isolate BLOCK PLACEMENT plus the dense-width
redistribution. Note I_G/hidden around 10-15 is deliberately unusual; pre-flight point 6's
`I_e/hidden >= ~0.5` rule constrains EXPERT width (1.33 and 5.73 here, both fine) and says nothing
about dense I_G.

### 27.3 An all-MoE trunk CANNOT be both iso-total and iso-active. Proved, not assumed.

A dense block has `active == total`. A top-k MoE block has `active == (k/K) * total`. So replacing a
dense trunk with MoE at FIXED total must LOSE active. Swept K_S in {4,8,16,32} x k_S in {1,2,4} x
I_S in {128..4096} through `audit_config` against `abl_AB` (6G6E deep, TOTAL 399.85 / ACTIVE 238.02 /
FLOPwt 238.02): **every iso-total sizing lands at 0.69-0.84x the reference ACTIVE (best 0.842), and
every iso-active sizing costs at least 1.19x TOTAL. There is no overlap.**

`abl_AC` takes iso-ACTIVE, because an arm 15.8% short on FLOPs repeats the defect that made the old
sandwich rows unreadable, and because no recurrence appears in either arm so `FLOPwt == ACTIVE` and
matching one matches both:

| arm | TOTAL | ACTIVE | FLOPwt | I_S/hidden | shape |
|---|---|---|---|---|---|
| `abl_AB` 6G6E deep | 399.85 | 238.02 | 238.02 | dense | GGGGGGEEEEEE |
| `abl_AC` 6S6E deep (NEW) | 475.40 | 238.07 | 238.07 | 1.00 | SSSSSSEEEEEE |

Trunk: K_S=8, top-4, I_S=1024. `k_S/K_S = 0.5` is only weakly sparse, which is the direct price of
iso-active: a K/k=16 trunk at iso-active would need TOTAL **1.53B**. The iso-TOTAL alternative that
keeps the 400M tier is K_S=4, k_S=2, I_S=1024 -> TOTAL 399.88, ACTIVE/FLOPwt 200.30 (0.842x).
**`intermediate_size` is PER-EXPERT for `mlp_type: MoE` but TOTAL across experts for the energy
blocks.** The 6G6S baseline's trunk sits at I_S/hidden = 0.28, inside the band this project has
repeatedly found fails.

### 27.4 6G6E deep had never been rerun with rnorm=none

`abl_H_400M_6G6E_deep` is the only deep no-recurrence energy arm, and it is still on
`routing_norm: zscore` -- the one 400M arm that never saw the largest knob effect we measured.
`abl_AB` is the rerun (a RERUN, not a resume: the parent completed its cosine decay to the LR floor,
so extending the schedule would re-raise the LR mid-run). Only deliberate differences from the
parent: `routing_norm` on all six energy blocks, the data path, and the names.

Kept at mbs 2 / ga 8, matching the parent that completed at a median **1.509 s/step (~26 h for
32.0B)**. mbs 4 / ga 4 would move tokens/call from the 8,192 losing column to the 16,384 winning one
at identical tokens/step and is probably much faster, but it raises activation memory on a
12-distinct-layer model and was not smoke tested. Worth a separate run; an OOM here costs more
than the hours saved.

### 27.5 Ditched per user instruction

`5G1x6E1G` (`abl_U`) and `5S1x6E1S` (`abl_Z2`) are block-POSITION variants, not true sandwiches, and
are reserved for the appendix or a reviewer question. `abl_Z2` was killed at step 430 and commented
out of `watchdog_jobs.conf` with a `#DITCHED-20260923` marker so the watchdog cannot resurrect it.
`abl_U` had already finished; its numbers stand as an appendix datapoint. Note `abl_U` is
byte-identical to `abl_R` on all three parameter axes, which is why it was never a capacity test.

### 27.6 abl_X and abl_Y named the owners' doomed data path

Found while auditing: **506 configs still name `/proj/datasets/granite-4-datasets-megatron-merged/`,
including the two live priority-1 arms `abl_X` and `abl_Y`.** The running processes would survive the
owners deleting it (open descriptors plus our hard links keep the inodes), but **the next LSF requeue
would fail to start**, and `abl_Y` has already been requeued 8 times.

Both were repointed to the shared FULL copy (4 paths each, 8 diff lines, verified weights still
0.35/0.35/0.15/0.15 and split 99,0.5,0.5). `data_cache_path` was deliberately left alone: fewer
moving parts on a resuming arm, and Megatron keys the blend index on the paths so it will build a
fresh entry in the existing cache dir anyway. Editing the config does not touch the running process,
which only re-reads it on requeue.

**Consequence to be honest about:** on the next requeue those arms rebuild the blend index and
therefore see a different sample ORDER from that point. At under one epoch on an identical corpus and
mixture that is statistically harmless, but it is not bit-exact. The alternative was losing the arm
outright, so the repoint strictly dominates.

### 27.7 Queue state after the reshuffle

`grp_ebm` is 32/32 and every one of those 32 GPUs is now a top-priority arm.

| arm | job | queue | priority | s/step | ETA |
|---|---|---|---|---|---|
| `abl_AA_134M_isoall` | 1885250 | grp_ebm | **0** iso-all sandwich | 0.135 | ~4.6 h |
| `abl_AB_400M_6G6E_rn` | 1885300 | grp_ebm | 6G6E rnorm=none | 2.44 early | 26-41 h |
| `abl_X_400M_rnorm` | 1876408 | grp_ebm | 1 Table 1 | 3.368 | 46.5 h |
| `abl_Y_400M_sandw` | 1876409 | grp_ebm | 1 Table 1 | 1.556 | 19.0 h |
| `abl_Z1_allmoe_hyb` | 1885299 | preemptable | lower | 0.357 | resumes at step 8000 |
| `abl_AC_6S6E_deep` | 1885332 | preemptable | 2 | - | PEND |

`abl_Z1` was moved off `grp_ebm` (killed at step 8670, resubmitted on preemptable, resumes from its
step-8000 checkpoint) to free the 8 GPUs that `abl_AB` now uses.

---

## 28. Was InfiniBand ever actually broken? The evidence says probably not (2026-09-23)

### 28.1 The claim, and why it does not hold up

`scripts/bsub/submit_train.sh` has exported `NCCL_IB_DISABLE=1` for every multi-node job since
commit `fd52703a` (**2026-09-18 21:48 UTC**), justified in a comment as: the cluster's InfiniBand
fails the first inter-node RDMA with `IBV_WC_RETRY_EXC_ERR(12)` -> `ncclRemoteError`, across 6+ HCAs,
7+ peers and 4 host pairs, and `NCCL_IB_TIMEOUT`/`NCCL_IB_RETRY_CNT` do not fix it.

**But `scale32B_boltz_sinkhorn` job 1701525 ran `DeviceMesh((pp=1, ddp=16, fsdp=1, tp=1))` -- 2 nodes
x 8 GPUs -- to step 58,260 with ZERO `IBV_WC_RETRY_EXC_ERR` and ZERO `ncclRemoteError`, and its log is
dated 2026-09-16, two days BEFORE `NCCL_IB_DISABLE` was introduced.** A second run, job 1667174,
reached step 34,170 on the same 16-GPU mesh, also clean. Note `fsdp=1` in both: those runs were
already `stage: 0`, so neither FSDP nor TCP was involved. So InfiniBand demonstrably carried a
58k-step 2-node job days before we concluded the fabric was broken.

HANDOFF §13.9 open decision 2 already recorded the doubt and never resolved it: *"Was the
`ncclRemoteError` ours? Raised, NOT investigated. `scale32B` ran 58k steps multi-node without it,
which argues cluster rather than code -- inference, not a test."*

### 28.2 The justification also conflates two different failures

The comment says `NCCL_IB_DISABLE=1` fixed arms that "stalled at step 0". **A stall at step 0 with no
NCCL error is the WEDGE signature**, which §13.3 attributes to `fused_experts` +
`repulsion_space: weight` reshaping a dim-0-sharded DTensor by expert. `ncclRemoteError` at
SeqNum=1 is the opposite: loud, and before training. If some of the arms credited to the IB switch
were actually wedging on weight-space repulsion, then disabling IB was given credit for a fix it did
not perform, and we have been paying TCP bandwidth on every inter-node all-reduce since.

**Why this costs real time:** at `stage: 0` there is no parameter sharding, so the FULL gradient is
all-reduced every step. At 400M that is ~1.9 GB of traffic per step and at 1B ~4 GB, and on a 4-node
job most of it crosses nodes.

**CORRECTION (same day, measured): the "10x bandwidth difference" I first wrote here was WRONG.**
The `NCCL_IB_DISABLE=1` path is NOT slow Ethernet. `ibtest_tcp`'s NCCL banner shows it using eight
IPoIB interfaces (`ibp26s0` ... `ibp220s0`, subnets 100.126.40-47.x) plus `bond1`, and
`/sys/class/net/*/speed` puts **`bond1` and every `ibp*` at 400,000 Mb/s**; only `bond0` is slower at
50,000. NCCL spread the traffic over sockets 0,1,4,6,9, i.e. IPoIB rails and `bond1`. So the links
were never the bottleneck and the transport runs over the same InfiniBand hardware, just through the
TCP/IP stack.

The real cost of disabling IB is therefore PROTOCOL overhead, not link rate: no GPUDirect RDMA, so
transfers bounce through host memory with CPU copies instead of going zero-copy NIC-to-GPU. That is
worth roughly 1.3-2.5x on the comms portion, not 10x. Measured decomposition at 2 nodes x 2 GPUs:
`ibtest_tcp` medians **1.084 s/step**, while `abl_AB` at the same FLOPwt does 65,536 tokens/rank in
1.380 s single-node, so 32,768 tokens/rank should cost ~0.69 s of compute. Inter-node overhead is
thus ~0.39 s/step, about **36%**, which bounds the best possible RDMA win at ~1.6x end-to-end and
makes 1.2-1.4x the realistic expectation. Worth having; not transformative. Do not quote a larger
figure.

Also note `NCCL_SOCKET_IFNAME` is already set to `ib,bond` BY THE CLUSTER ENVIRONMENT, not by our
submitter.

### 28.3 The test now running

`ALLOW_IB=1` was added to `submit_train.sh` as an opt-in escape hatch; the default stays disabled,
because arms launched without it did sit dead at step 0 and that is not yet explained.

Two jobs, identical config, 2 nodes x 4 GPUs each (`GPN_FORCE=4` with `GPUS=8`, so it is genuinely
inter-node on only 8 GPUs), `NCCL_DEBUG=INFO` on both so the chosen transport is visible in the log:

| job | arm | transport |
|---|---|---|
| 1885615 | `ibtest_tcp` | `NCCL_IB_DISABLE=1` (current default) |
| 1885616 | `ibtest_ib` | `ALLOW_IB=1` |

Config `configs/iclr_26/ablations/_ibtest_6S6E_{tcp,ib}.yml`: the 400M 6S6E all-MoE arm, 500 steps,
no checkpoints. `sparse_start_step: 300`, so the probe covers both the dense and the sparse regime.
Compare the median s/step over the last 20 steps, and grep for `NET/IB` vs `NET/Socket`.

**Read the result carefully.** A single multi-node s/step number is not admissible on its own: the
same config has measured 2.45-6.81 s/step across host placements, a 2.7x spread from placement alone.
If IB wins by less than ~2x, repeat on other placements before believing it. If the IB arm dies at
the first collective with `IBV_WC_RETRY_EXC_ERR`, that is the documented fault reproducing and the
default was right.

Both jobs are PEND: the preemptable GPU pool is full, and `span[ptile=1]` across 2 hosts is harder to
place than a single-node job.

### 28.4 1B bharat configs repaired

All six were updated. Two findings beyond the data path:

1. **`bharat_1B_16S_switch_baseline.yml` and `bharat_1B_8S8E_boltz.yml` were reading the BIASED
   18.6%-prefix subset** (`granite-4-cmix-subset`), while `bharat_1B_18L_allMoE_*` and the
   `fallback/` pair read the owners' full corpus. The subset prefix is **~0.47 nats easier**, so that
   Switch-vs-energy pair at 1B was not comparable, in the direction that flatters whichever arm sat
   on the subset. Both now read the shared FULL copy.
2. **`bharat_1B_8S8E_boltz.yml` would have CRASHED AT STARTUP.** It sets `proxy_rank: 16` with
   `fused_experts: false`, and `energy_ff.py:1090` asserts
   `not (proxy_rank > 0 and not fused_experts)` -- *"the looped expert path never runs the proxy, so
   the proxy would silently never be trained or measured"*. Set `fused_experts: true` and
   `sparse_forward: true`, which also makes it match `bharat_1B_18L_allMoE_boltz`.

Also `routing_norm: zscore -> none` on all 30 energy blocks across the three energy configs (18 + 8 +
4), and `data_cache_path` / `tokenizer_name` moved to the shared tree.

**Left alone deliberately: `mbs 2 / ga 8`.** That puts tokens/call at 8,192, which is the column
where `sparse_forward` LOSES (0.41-0.49x); 16,384 is where it wins 2.1-2.4x. `mbs 4 / ga 4` keeps
tokens/step identical and would likely be much faster, but at 1B (d=1536, 18 layers) it raises
activation memory and was not smoke tested. Worth one probe before a long launch.

**Budget warning for whoever launches these:** all six are `61035 steps x mbs 2 x ga 8 x 4096`, so
tokens/step depends entirely on the GPU count. 8 GPUs gives 32.0B, 16 gives 64.0B, 32 gives 128.0B.
Per §19.9, the 1B tier needs 16 or 32 GPUs, never 24: 524,288 tok/step is 128 sequences and 128/24 is
not an integer.

### 28.5 ROOT CAUSE CANDIDATE: 10 HCAs on TWO IB SUBNETS, and NCCL was pairing across them

Diagnosed 2026-09-23 with `ibstat` / `ibv_devinfo` on a compute node, no GPU job required.

**The fabric is not down.** All ten HCAs are ConnectX-7 (MT4129), every port `State: Active`,
`Physical state: LinkUp`, `Rate: 400`. But they are managed by TWO DIFFERENT SUBNET MANAGERS:

| HCAs | SM lid | IPoIB interface | role |
|---|---|---|---|
| `mlx5_0,2,3,4,5,7,8,9` (8) | **1923** | yes (`ibp*s0`, 100.126.40-47.x) | compute fabric |
| `mlx5_1`, `mlx5_6` (2) | **1** | **none** | storage / management |

Two independent confirmations of the same 8/2 split: the SM lid grouping above, and NCCL's own
banner in `ibtest_tcp`, which enumerates exactly EIGHT `ibp*` IPoIB interfaces and no interface for
`mlx5_1`/`mlx5_6`. `nvidia-smi topo -m` corroborates the pairing: NIC0-NIC1 are `PIX` (same PCIe
switch) and NIC5-NIC6 are `PIX`, so each PCIe complex carries one storage rail next to a compute rail.

**Why this produces exactly the reported error.** If NCCL enumerates all ten rails and pairs a rail on
subnet 1923 with a peer's rail on subnet 1, the two cannot route to each other, the remote QP never
answers, and the completion comes back `IBV_WC_RETRY_EXC_ERR(12)` -> `ncclRemoteError`. That is a
multi-rail SELECTION fault, not a broken fabric. It also explains the detail in the original comment
that otherwise reads as damning: "across 6+ HCAs, 7+ peers". Of course it failed across many HCAs,
because NCCL was trying rails that cannot reach one another. And it explains why
`NCCL_IB_TIMEOUT`/`NCCL_IB_RETRY_CNT` did nothing: retrying an unroutable path does not help.

**Fix, in `submit_train.sh` under `ALLOW_IB=1`:** select `NCCL_IB_HCA` AT RUNTIME per node as the set
of HCAs on the majority SM lid. Computed rather than hardcoded so it survives different HCA naming or
counts on other host classes.

### 28.6 Experiment design: three arms, because one job cannot separate the two hypotheses

| job | arm | transport | what it tests |
|---|---|---|---|
| 1885784 | `ibtest_tcp` | `NCCL_IB_DISABLE=1` | baseline. **RESULT: 1.0853 s/step** (2 nodes x 2 GPUs, 131,072 tok/step, median of all steps after warmup; the last-20 median is 1.0866, so the arm is very stable) |
| 1885785 | `ibtest_ib` | IB, ALL 10 rails (submitted BEFORE the fix, deliberately) | if the cross-subnet pairing is the cause, this reproduces `IBV_WC_RETRY_EXC_ERR` |
| 1885815 | `ibtest_ibhca` | IB, compute rails only | should run, and beat 1.0853 |
| 1886040 | `ibtest_ibmin` | IB, compute rails only, **2 nodes x 1 GPU**, 120 steps | pure CONNECTIVITY test. Smallest ask on the cluster so it places soonest, and with one GPU per node 100% of the reduction crosses the boundary. Answers "does RDMA connect" without needing a matched footprint |

`ibtest_ib` being pre-fix is the control that makes the experiment interpretable: if it fails and
`ibtest_ibhca` succeeds, the fabric was never the problem. If BOTH fail, the original conclusion was
right and `NCCL_IB_DISABLE=1` stays.

**Caveat on the timing comparison:** `sparse_start_step: 300`, so the 1.0853 baseline above is the
DENSE regime. Compare like with like, and read the sparse regime separately from steps 300-500. And a
single multi-node s/step is not admissible alone: the same config has measured 2.45-6.81 s/step across
placements, a 2.7x spread from placement. The 2x2 arms must be compared on comparable host pairs, or
the spread reported.

### 28.7 RESULT: InfiniBand WORKS. The "broken fabric" conclusion was wrong.

Measured 2026-09-23. Median s/step, split at `sparse_start_step: 300`, from
`scripts/ib_probe_report.sh`. "err" counts GENUINE NCCL errors only (see the trap below).

| arm | hosts | transport | dense (21-299) | sparse (300+) | err |
|---|---|---|---|---|---|
| `ibtest_tcp` | p6-r07-n4:p3-r18-n3 | NET/Socket | 1.0883 (n=27) | 0.8938 (n=21) | 0 |
| `ibtest_ib` (ALL 10 rails) | p2-r28-n3:p4-r30-n3 | **NET/IB** | 1.2744 (n=16) | - | **0** |
| `ibtest_ibhca` (8 rails) | p5-r10-n2:p2-r14-n4 | **NET/IB** | 0.9247 (n=27) | 0.7706 (n=4) | **0** |
| `ibtest_ibmin` (8 rails, 2x1 GPU) | p6-r07-n4:p6-r09-n4 | **NET/IB** | 0.9384 (n=10) | - | **0** |

**ESTABLISHED: native RDMA works on this cluster.** Three arms negotiated `NET/IB` over the mlx5
rails, trained, and produced ZERO genuine NCCL errors; `ibtest_ibmin` ran 120 steps to
`Successfully completed`. So `NCCL_IB_DISABLE=1`, which has forced TCP on every multi-node job since
2026-09-18, was not necessary. `scale32B`'s clean 58,260-step 2-node run on 2026-09-16 was not a
fluke.

**MY RAIL-RESTRICTION HYPOTHESIS IS NOT THE EXPLANATION.** `ibtest_ib` ran with ALL TEN rails, no
`NCCL_IB_HCA` at all, and connected on `NET/IB` with zero errors. So the two-subnet cross-pairing
story (28.5) is NOT what makes IB work here. The original `IBV_WC_RETRY_EXC_ERR` looks TRANSIENT or
specific to the hosts drawn that day, which is exactly what 13.9 open decision 2 suspected and never
tested. The runtime `NCCL_IB_HCA` selection is harmless hygiene, but do not describe it as the fix.

**The ~18% speed edge is NOT YET ADMISSIBLE.** 1.0883 -> 0.9247 dense is 1.18x, but on DIFFERENT host
pairs, and this project has measured the same config at 2.45-6.81 s/step across placements, a 2.7x
spread that is larger than the effect. `ibtest_ib`'s 1.2744, SLOWER than TCP on IB, makes the point:
that is its host pair, not its transport. A placement-controlled pair (`ibpair_tcp` 1886164 /
`ibpair_ib` 1886165, both PINNED to p6-r07-n4:p6-r09-n4 via the new `FORCE_HOSTS`, 400 steps, 2 nodes
x 1 GPU so 100% of the reduction crosses the boundary) is running to settle it.

**Sparsity confirmed on both transports**: TCP 1.0883 -> 0.8938 and IB 0.9247 -> 0.7706 past step 300,
consistent with `sparse_forward` winning at tokens/call 16,384.

### 28.8 Three process traps this exposed, all self-inflicted

1. **NEVER grep bare `ncclRemoteError` / `IBV_WC_RETRY_EXC_ERR` in a job log.** `submit_train.sh`'s own
   comments AND its `ALLOW_IB` warning echo contain those strings, and the generated job script is
   echoed into the log. A bare grep reported an "IB-FAULT" on three healthy arms and then "4 errors"
   on a job that had `Successfully completed`. I announced the hypothesis "refuted" on the strength of
   that false positive before reading the log. Require the real NCCL line format
   (`host:pid:tid` or `NCCL WARN`) -- encoded in `scripts/ib_probe_report.sh`.
2. **NCCL writes its banner and errors to STDOUT; step lines go to STDERR.** A monitor watching only
   stderr sees no transport and no NCCL error. This is the mirror image of the documented trap where
   grepping stdout for step lines makes a healthy job look wedged. Scan both.
3. **THE EVAL AUTOMATION SWEEPS UP THROWAWAY RUNS.** `sparse_eval_followup.sh` globs
   `results/**/unsharded*/config.json`, so the 120-500 step transport probes under `results/_smoke/`
   were auto-unsharded and had FOUR eval jobs launched against them. Killed, and the glob now skips
   `_smoke` / `ibtest` / `ibpair`. The real hazard is not the wasted GPU: an eval of a 120-step model
   can land in a results table and be read as an arm.

Also: `ibtest_tcp` briefly went `SSUSP`. Suspension distorts the step times around it -- its
last-15 median read 0.8898 while its true dense median was 1.0883, because the window had crossed
into the sparse regime AND been perturbed. Always split regimes and check `STAT` before quoting.

### 28.9 FINAL: native IB is 1.40x dense / 1.62x SPARSE, placement-controlled

Both transports on the SAME host pair `p6-r07-n4 : p6-r09-n4`, same config, 2 nodes, the 400M 6S6E
model, 400 steps each. Median s/step; lower is better. Zero genuine NCCL errors in either arm.

| regime | TCP (`NET/Socket`) | native IB (`NET/IB`) | speedup |
|---|---|---|---|
| dense (steps 21-299) | 1.2861 (n=27) | 0.9160 (n=27) | **1.40x** |
| **sparse (steps 300+)** | 1.1448 (n=11) | 0.7069 (n=11) | **1.62x** |

**QUOTE THE SPARSE ROW.** `sparse_start_step` is 300 out of 61035-122070 steps, so a production run is
sparse for over 99% of its life. 1.40x is the dense-phase number and understates the real gain.

**The mechanism is consistent, which is why this is believable and not just a lucky pair.** The
ABSOLUTE IB saving is ~0.37-0.44 s/step in BOTH regimes. That is expected: at `stage: 0` the gradient
all-reduce volume does not change with sparsity, so the comms saving is constant, while sparsity cuts
compute -- so the same saving is a larger fraction of a shorter step. Decomposed against ~0.69 s of
expected compute, inter-node overhead is 0.596 s on TCP (46% of the step) against 0.226 s on IB (25%),
i.e. **RDMA removes ~62% of the inter-node cost**, comfortably inside the ~1.56x ceiling implied by the
earlier 36%-overhead estimate.

**THE CO-TENANCY WORRY WAS UNFOUNDED -- I raised it and then disproved it.** The two arms ran
CONCURRENTLY on the same two hosts, so I flagged that they shared host CPU and NICs and that TCP,
being CPU-bound on protocol processing, would be unfairly penalised. When `ibpair_ib` finished,
`ibpair_tcp` kept running alone, which splits its own steps into contended and uncontended:
1.2861 (n=26) with the co-tenant versus 1.3418 (n=3) without. **Co-tenancy cost TCP 0.96x, i.e.
nothing** (it was marginally slower alone, which is noise at n=3). So the concurrent same-host design
is sound and the 1.40x/1.62x stand.

**Why the cross-pair numbers were confusing: the BASELINE moves with placement.** TCP measured 1.0883
on `p6-r07-n4:p3-r18-n3` but 1.2861 on `p6-r07-n4:p6-r09-n4`, a 1.18x spread in TCP alone. And one IB
arm (`ibtest_ib`, `p2-r28-n3:p4-r30-n3`) came in at 1.2613 -- ON IB AND SLOWER THAN TCP-on-a-good-pair.
That is why cross-pair readings ranged 1.17-1.39x and why only the matched pair is admissible. It also
vindicates the standing rule: no multi-node s/step claim without placement control.

**Limits, so this is not over-trusted:** ONE host pair, and 2 nodes x 2 GPUs. At 4 nodes x 8 GPUs the
comms FRACTION differs -- more of the reduction is absorbed by intra-node NVLink, but there are more
node boundaries. Per-rank tokens (32,768) are identical to the 4-node config, so the compute
arithmetic carries over; the ratio does not. Measure it on the real run.

**Action taken:** `configs/iclr_26/bharat/bharat_400M_6S6E_deep_allmoe_128B_4node.yml` now says to
launch with `ALLOW_IB=1` and its estimate is revised from 41-58 h to **27-44 h**. The submitter's
DEFAULT stays `NCCL_IB_DISABLE=1`: the original 2-of-4 startup failures were real and documented, and
today only shows IB working on the hosts we happened to draw. A first-collective `ncclRemoteError` is
a resubmit (the watchdog self-resubmits); a SILENT stall at step 0 with zero NCCL errors is the
weight-space repulsion wedge instead and resubmitting will not help.

---

## 29. Pivot to FineWeb-Edu (2026-09-25)

### 29.1 Why the pivot

The Megatron cmix results (§1-28) showed every Boltzmann variant trailing Switch at d=768.
But we discovered the comparison was confounded by five simultaneous changes between the old
champion (h1 680M, w1w2 dense, legacy code) and the ICLR grid (hopfield sparse, new code).
A fast FineWeb-Edu sweep was designed to isolate each factor.

Data: nemotron-cc web-hq p2_0 shard (10.4B tokens, granite-4.0 tiktoken, megatron-indexed).
NOT FineWeb-Edu itself (incompatible .idx format) — the name is historical.
LR: 1e-3 (µP-derived, optimal from d=384 to d=768).
Schedule: cosine decay, no constant phase.
GSM8K: skipped (too small at this scale; separate benchmark if needed).
MMLU: reported separately from Avg11 (same convention as the megatron runs).

### 29.2 Scale 1 results (d=768, 4.0B tokens, COMPLETE)

All Avg11 = 11-task accuracy mean (pp, higher better). PPL = wikitext word-level (lower better).

| arm | arch | total | active | FLOPwt | lm_loss | Avg11 | MMLU | PPL |
|---|---|---|---|---|---|---|---|---|
| 12G baseline ★ | 12G | 162M | 162M | 162M | 3.164 | 43.95 | 26.3 | 39.2 |
| Switch 6G1x6S K=8 ★ | 6G1x6S | 163M | 132M | 196M | 3.216 | 44.72 | 24.1 | 42.1 |
| Boltz 6G1x6E K=8 | 6G1x6E | 162M | 141M | 251M | 3.283 | 43.96 | 25.1 | 45.4 |
| Boltz 6G1x6E K=16 (isoFLOP) | 6G1x6E | 168M | 127M | 162M | 3.322 | 42.61 | 24.3 | 47.9 |
| w1w2 dense K=4 (new code) | 6G1x6E | 133M | 133M | — | 3.236 | 43.96 | 24.1 | 43.4 |
| legacy w1w2 dense K=4 | 6G1x6E | 133M | 133M | — | 3.220 | 43.62 | 23.4 | 42.5 |

**Key findings:**
1. Switch leads Boltzmann by 0.76pp Avg11 at iso-total (44.72 vs 43.96), but Boltzmann uses
   28% more FLOPs (251 vs 196M) due to the 6× recurrence of the energy block.
2. Boltzmann matches the 12G baseline on Avg11 (43.96 vs 43.95) with 13% fewer active params.
3. At iso-FLOP (K=16, FLOPwt=162M = 12G baseline), Boltzmann drops to 42.61 — 1.3pp behind.
4. Legacy code ≈ new code (43.62 vs 43.96, within single-seed noise). Safe to use new code.
5. psd_anti > unconstrained by ~0.07 nats (probes, not in table).
6. sparse_explore: 0 gives 2-3× wall-clock speedup with no quality loss.

### 29.3 Shared computation: Boltzmann routing is free

For w1w2 Boltzmann MoE, the routing energy E_k = h · term1_k requires the same 3 GEMMs
(W1·h, W2·h, phi(W1h)@W2^T) that produce the expert output gradient. So 75% of each
selected expert's arithmetic is shared with routing. The incremental cost of Boltzmann
routing vs a "dumb" expert-forward is one additional GEMM (term2 back-projection, 25%) plus
a dot product + softmax (negligible). A separate learned gate (as in Switch) adds O(dK) on
top of the expert forward — a DIFFERENT cost, not shared.

### 29.4 Speed optimizations applied

| change | speedup | mechanism |
|---|---|---|
| 8 GPUs (ga 8→4) | ~2× | half the sequential microbatch forwards per step |
| sparse_explore 2→0 | ~35% | removes 2 redundant candidate evaluations per token |
| compile_helpers skip | ∞ (removes 10-30 min hang) | skips ninja rebuild when .so loaded |
| combined | 3.1× (4.5→1.45 s/step at d=768 K=8) | |

### 29.5 Alignment measurement (6S6E d=768, step 4000)

Consecutive-layer cosine similarity in the G1 deep model (6 Switch + 6 Boltzmann, no recurrence):
- Energy attention: 0.16-0.25 (dominant signal, drives layer-to-layer correlation)
- FF/MoE: <0.07 (much smaller)
- Full layer delta: 0.17-0.44 (both signals combined)
- Weight alignment: near zero — output alignment is emergent, not from shared parameters.
- The alignment regularizer (G2) HURTS by ~0.06 nats at d=768. Not pursuing further.

Root cause of initial "zero attention" measurement: RoPE cos/sin buffers are non-persistent
and were zeroed after meta-device loading. Fix: call model.transformer.rope.reset_parameters()
after loading. Documented in measure_alignment_v3_20260925.py.

### 29.6 Running and planned

Scale 2 (d=1024) and Scale 3 (d=1280) are running. See `configs/iclr_26/priority.md`.

### 29.7 Token calculation formula (AUTHORITATIVE)

```
tokens = seq_length × micro_batch_size × gradient_accumulation × total_GPUs × num_train_steps
```

**GPUS IS NOT IN THE CONFIG.** Use `experiments/boltzmann-moe/scripts/compute_tokens.sh`.

Found 2026-09-25: G3 (6S6E d=768) and G4 (6S6E d=1024) had ga=4 and ga=2 respectively
when they should have had ga=8 and ga=4, giving them HALF the intended tokens (2.0B and
3.8B instead of 4.0B and 7.6B). All d=1280 arms ran at ga=2 = 5.6B tokens = 0.8-0.9×
Chinchilla (undertrained).

Corrected 6S6E arms (G6/G7/G8) submitted with:
- G6 d=768:  4GPU, ga=8, active=132M (iso-active with Switch E3), 4.0B tokens, 1.5× Chin
- G7 d=1024: 8GPU, ga=4, active=225M (iso-active with Switch F3), 7.6B tokens, 1.7× Chin
- G8 d=1280: 8GPU, ga=2, active=315M (iso-active with Switch H3), 5.6B tokens, 0.9× Chin

### 29.8 Final results (2026-09-25 22:30 UTC)

19 arms evaluated across d=768/1024/1280. All param counts from `audit_config`.
Token formula verified: `seq × mbs × ga × GPUs × steps` (script: `compute_tokens.sh`).

**Cross-scale Avg11 (iso-active within each scale):**

| arm | d=768 | d=1024 | d=1280 |
|---|---|---|---|
| Switch MoE (6G1x6S) † | **44.72** | **46.01** | **46.89** |
| 12-layer dense (12G) † | 43.95 | 45.56 | 46.16 |
| 12S all-Switch (12S) † | — | 45.51 | — |
| Boltz-MoE deep (6S6E) | 43.47 | 45.07 | **46.17** |
| Boltz-MoE rec (6G1x6E) | 43.96 | 44.73 | 45.45 |

**The headline:** 6S6E deep matches the 12G baseline at d=1280 (46.17 vs 46.16, Δ=+0.01pp)
and trails Switch by only 0.72pp while using 2.1× fewer FLOPs (315M vs 661M). The 6S6E deep
architecture is strictly better than the recurrent 6G1x6E at d≥1024: more quality, fewer FLOPs.

**Token efficiency:** Switch reaches Boltzmann's final loss at 53% of Boltzmann's tokens.
12G reaches EGPT's at 33%. The energy model is ~2× less token-efficient than Switch but the
gap is a constant factor, not a scaling barrier.

**Corrected arms (G6/G7/G8/G81):** the original G3/G4 had half tokens due to a ga config bug
(documented in §29.7). G5 had mismatched active params (355M vs 315M target). All three
corrected arms are iso-active with Switch at their scale.

**New baseline (S1):** 12S all-Switch at d=1024, iso-active 225M, 7.6B tokens. Avg11=45.51,
between 12G (45.56) and Switch 6G1x6S (46.01). This is the fairest deep-Switch-vs-deep-energy
comparison: same depth (12 layers), same tokens, same active params, no recurrence on either
side. The energy model (6S6E, 45.07) trails by 0.44pp.
