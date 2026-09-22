# CRITIC_LOG — paper-critic agent, ICLR Boltzmann-MoE

Append only. One dated section per exchange. Read before re-deriving anything.

## 2026-09-21 — session 1: Q1 (likelihood vs partition function), Q2, Q3, Q4 + rejection list

### Verified numerically this session (float64, CPU, `/tmp/critic/chk.py`, `chk2.py`)
- Hopfield FF branch with `gelu_grad_method: sigmoid`: cos(forward, grad_h E) = **0.9968**
  (sigma=0.02), 0.9978 (0.05), 0.9998 (0.2). NOT 1. `sec/debug.tex`'s row
  `sqrt_consistent | any | sqrt(Ie) | 1` is wrong in the cos column AND in "any": the ratio is
  sqrt(Ie) only because the sigmoid phi' is half-magnitude; with `erf_exact` it is 2*sqrt(Ie).
  The rest of that table IS internally consistent once the 1/2 is accounted for
  (sqrt_consistent = 4/sqrt(Ie); 4/sqrt(Ie) / (2/Ie) = 2 sqrt(Ie) = 133.9; x 1/2 = 66.93).
- `sec/appendix.tex` app:expert-forms states phi' = 1/2 sigma(sqrt(2/pi) z); `sec/debug.tex`
  states 1/2 sigma(1.702 u). DIFFERENT surrogates, different cosines (0.9968 vs 0.9991).
  debug.tex's quoted 0.99682862 matches the sqrt(2/pi) form, so debug.tex's "1.702" is the typo.
- vMF normaliser C_d(kappa): log C is FLAT for kappa << nu = d_h/2 - 1 and collapses for
  kappa >~ nu. At d_h=64 (nu=31): log C(kappa) - log C(1) = -0.12 (k=4), -0.49 (k=8),
  -1.94 (k=16), -7.24 (k=32), -24.2 (k=64). Typical scale under standard init
  (||g||=sqrt(d), W~N(0,1/d)) is kappa ~ sqrt(d_h) = 8, i.e. kappa/nu ~ 0.26 -> FLAT.
  A 2x key-norm spread costs 1.8 nats of omitted logit bias (factor 6 in mixture weight);
  4x costs 7.2 nats (factor 1.4e3); 8x costs 24 nats (3e10). => the vMF/KDE reading is
  approximately exact in the regime the model operates in, and fails hard for attention sinks.

### Verified from code / configs (`configs/cmix/cmix_134M_hybrid_32B_sparse.yml` and
### `configs/iclr_26/scaling/cmix_400M_hybrid_sparse.yml`, checkpoint `config.json`)
- `energy_proj_type: **psd_anti**` in the 32B headline arms, NOT `dual_unconstrained`.
  `PSDAntisymmetricProjection` (lm_engine/hf_models/models/energy/layer.py:73) is
  Pi = S S^T + (A - A^T) applied as a SINGLE map to `attn_out + scale_ff*ffwd_out`, with a
  docstring proving Edot = -||S^T grad E||^2 <= 0. So these arms DO have a descent guarantee.
  app:expert-forms' claim that every reported arm is `dual_unconstrained` is FALSE for the 32B wave.
- `energy_unified_grad: False`, `energy_self_k_diag: False`, `energy_apply_rayleigh: False`
  in every paper arm (defaults; absent from all configs under configs/cmix and configs/iclr_26).
  => attention branch = QUERY-PATH partial only; the key-path term of grad_h E^AT requires
  FUTURE positions and is causally unavailable. Not mentioned anywhere in the paper.
  => omitting Rayleigh means the returned vector is grad_g E, not grad_h E, and the RMSNorm
  tangent projector breaks the psd_anti descent guarantee (restored if Rayleigh is ON).
- `routing_norm: zscore` in the headline arms => p_k = softmax(beta (E_k - Ebar)/std(E)),
  which is NOT softmax(beta E_k) and so eq:free-energy-grad does not hold for it.
- `renormalize_topk: true` in the headline arms, contradicting app:routing ("Truncation does
  not renormalise ... sum p_k < 1 by design") and making tab:frontier's "renorm." row not an
  ablation but the shipped default.
- `repulsion_coef: 0.1` AND `proxy_loss_coef: 0.01` => TWO auxiliary loss terms in every
  reported arm, against the abstract/intro/related claim of "no auxiliary term in the loss".
- `proxy_rank: 16, proxy_kind: subspace, proxy_out_dim: 512, proxy_init: svd` => the sparse
  arms select experts with a LEARNED, PARAMETRIZED, separately-distilled router.

### Numbers: stale vs current
- Intro's 43.83/43.83, 44.58/44.43, 1.33pp, 0.71pp are the **7.86B** wave (tab:frontier) —
  internally consistent with the main body but a QUARTER of the appendix's 32B budget.
- Intro's "12.3-12.8 of 16, share 0.17-0.18" is superseded by sec:balance itself
  (15.2-16.0 of 16, share 0.063-0.130). Pre-sign-fix, pre-mu.
- 32B (tab:status / HANDOFF 16.2): hybrid 6G1x6E **44.82** is behind abl_E no-MoE 45.87,
  abl_D GPT-only iso-total 45.55, abl_B Switch FLOP-matched 45.43, abl_F GPT-only iso-active
  45.01, AND the unmatched Switch 6G1S 44.87. It is the 7th of 11 arms.
- Margin to the FLOP-matched Switch SHRANK from 1.33pp (7.86B) to 0.61pp (32B). Favourable trend,
  currently unstated.

### Open after this session
- Q1 verdict delivered: adopt "free energy = -beta^-1 log Z over a FINITE LATENT index set";
  the density normaliser over representation space is a separate, intractable, g-independent
  Z_theta that the paper never needs and must not claim. "likelihood function for tokens" must go.
- Q4: the honest phrasing needs FOUR caveats (lambda=66.93, cos 0.9968, query-path-only,
  zscore routing) but psd_anti RESCUES the descent claim for the 32B arms.
- Not yet checked: sign of the learned `scale_ff` in the trained hybrid checkpoint. If negative
  in any block, the block ASCENDS E^FF. One safetensors read; probe crashed on bf16, retry with
  framework="pt".

### Measured on trained checkpoints (added later in session 1)
- `scale_ff` is **POSITIVE in every energy block of every arm probed** (3.6-8.0; 8.0 in the
  134M hybrid `transformer.h.6`, 6.5-7.1 in the 1B 8G4E at 32B). So the sign of the FF channel
  never inverts and the psd_anti descent guarantee holds sign-wise. GOOD NEWS.
- Combined with `hopfield_grad_scale: sqrt_consistent` (66.93x), the **measured relative weight
  of the two free energies in the 134M hybrid is lambda_FF = 66.93 x 8.0 = ~535**, not 1 as
  eq:total implies. eq:total must carry an explicit lambda_FF and report it.
- `SS^T` smallest eigenvalue is ~0 (-2e-7, bf16 round-off) by Marchenko-Pastur construction for
  a square S, so Pi is near-singular: Edot <= 0, NOT < 0. "strict energy descent" in the
  PSDAntisymmetricProjection docstring should be "non-increasing".
- **Concentration measured on the trained 134M hybrid energy block** (`transformer.h.6.attn.c_attn`,
  shape (1536,768) = Q and K blocks only, no V): d_h=64, H=12, nu = d_h/2-1 = **31**.
  Per-head rms ||W_h g|| at ||g||=sqrt(d): Q 2.52-3.87 (med 2.81), K 3.40-4.54 (med 3.70).
  => **kappa = ||q|| ||k|| / sqrt(d_h) ~ 1.1-2.2, median 1.3** — versus nu = 31.
  At kappa=2 the omitted vMF log-normaliser has moved only **0.023 nats**. So the KDE reading is
  exact to <0.03 nats of logit ON THE WEIGHTS. Caveat: rms over isotropic g, not real hidden
  states; outlier/sink keys can be 10-100x larger and would move kappa into the region where
  log C varies. ONE forward pass on real data settles it — do that before the claim ships.

### Q1 resolution (the convention to adopt)
TWO NESTED NORMALISERS, and the paper conflates them via the `\propto` in eq:moe-density/eq:kde.
1. `Z(g) = sum_s exp(-beta E_s(g))` — sum over a FINITE LATENT index set (K experts / A context
   tokens). It is a MARGINALISATION, not a normalisation. Function of g. Computed exactly.
   `F = -beta^-1 log Z` is the free energy = the UNNORMALISED negative log-density of g.
2. `Zcal_theta = int_{S^{d-1}} exp(-beta F(g)) dg` — the normaliser over representation space.
   Intractable, theta-dependent, g-INDEPENDENT. This is the Z an EBM cannot compute, and the
   paper still cannot compute it. It drops out of grad_g (all the forward pass uses) and out of
   every log-density DIFFERENCE. Finite only because the sphere is compact — the normalisation
   of g is load-bearing here, not conventional.
So E=-log P and F=-log Z are the SAME KIND of object (both unnormalised neg-log-densities);
they are not the same Z. The paper has not escaped the partition function, it has computed a
DIFFERENT, tractable one.
THE STRONG CLAIM THAT IS EXACT AND NEEDS NO NORMALISER:
  p_s(g) = exp(-beta E_s(g))/Z(g) = p(s|g) — the EXACT POSTERIOR over the discrete latent.
  Zcal_theta cancels. A learned gate approximates this; ours computes it.
  And eq:free-energy-grad is FISHER'S IDENTITY in g (grad of a log-marginal = posterior-expected
  grad of the log-joint). Name it; it upgrades "nobody chose the softmax" to a cited identity.
"LIKELIHOOD FUNCTION" IS INDEFENSIBLE, three independent strikes:
  (a) a likelihood is a function of theta at fixed data; p here is a function of g at fixed theta;
  (b) g is a hidden representation, not data — there is no term whose exp integrates to 1 over
      the vocabulary;
  (c) IT IS NOT DISTINGUISHING: a GPT's softmax head already gives an exactly normalised,
      explicitly parametrised, samplable token likelihood. The abstract's headline claim
      describes GPT-2. THIS IS THE SINGLE MOST DANGEROUS SENTENCE IN THE PAPER.
WHAT IS DISTINGUISHING: an explicit SCALAR ENERGY OVER REPRESENTATIONS whose gradient IS the
forward pass; the mixture weights are exact posteriors, not learned gates; and attention and the
FFN are the SAME object differing only in the index set.
WHAT THE `\propto` HIDES:
  eq:moe-density: 1/Zcal_theta. theta-dependent, g-independent, intractable. A genuine constant.
  eq:kde: NOT a constant at all. (i) the vMF per-kernel factor C_dh(kappa_AB), which is INSIDE
  the sum and depends on A and B — a model discrepancy, not a normaliser; (ii) 1/|C_A|, a genuine
  constant that varies with POSITION. So softmax attention is an exact vMF KDE iff C_dh(kappa_AB)
  is constant over the realised kappas — measured kappa 1.1-2.2 vs nu=31, so YES to <0.03 nats.

### PRIOR ART (delegated search, verified against sources) — THIS IS NOW THE #1 PROBLEM
`boltz_moe.bib` (61 entries) has **ZERO** score-matching citations and **ZERO** Sinkhorn/OT-routing
citations. Claims 3, 4, 5 currently read as unattributed.

| claim | status | most dangerous citation |
|---|---|---|
| attention = NLL of a vMF KDE on context tokens | PRE-EMPTED in substance | **Nguyen, Pham, Nguyen, Nguyen, Osher, Ho, "FourierFormer", NeurIPS 2022, arXiv:2206.00206** — writes the key-KDE `p_sigma(k) = (1/N) sum_j phi_sigma(k - k_j)` as an EQUATION and says attention follows from unnormalised Gaussian kernels + a GMM assumption on queries. Also **Tsai et al., "Transformer Dissection", EMNLP 2019, arXiv:1908.11775** (priority for attention-as-kernel-smoothing); **Han et al., NeurIPS 2023, arXiv:2210.05794** |
| FFN energy = log-sum-exp over stored patterns; grad = softmax mixture | PRE-EMPTED as mathematics | **Hu, Zou, Xu, "Hyper-SET", ICLR 2026, arXiv:2502.11646 — Table 20 "Softmax FF" row** gives our exact `E_FF = -sum_i log sum_m exp(d_m^T x_i)` and `-grad E_FF = D softmax(D^T X)`. Plus `ramsauer2020hopfield` (already cited twice) |
| score blind to relative mode weights | ALREADY IN THE LITERATURE | **Song & Ermon, NeurIPS 2019, arXiv:1907.05600, Sec 3.2.2** — verbatim: disjoint-support mixture `pi p1 + (1-pi) p2`, "the score does not depend on pi", Langevin "will not be able to correctly recover the relative weights". Also **Wenliang & Kanagawa, arXiv:2008.10087, "Blindness of score-based methods to isolated components and mixing proportions"** (the title IS the claim); **Hyvarinen 2005 JMLR 6:695 Thm 2 requires p0 > 0 everywhere**; **Koehler, Heckett, Risteski, ICLR 2023, arXiv:2210.00726** (isoperimetry) |
| energy parametrisation preferable to score | **REFUTED — and my prior was WRONG** | **Salimans & Ho, "Should EBMs model the energy or the score?", ICLR 2021 EBM workshop, OpenReview 9AS-TF2jRNb** concluded **PARITY** (FID 6.8 energy vs 6.5 unconstrained, same U-net) and said to stop arguing about the formalism. Bridge citation that does what we want: **Schroder, Ou, Lim, Li, Vollmer, Duncan, "Energy Discrepancies", NeurIPS 2023, arXiv:2307.06431** |
| balance = Sinkhorn dual / chemical potential | ALREADY IN THE LITERATURE | **Clark et al., "Unified Scaling Laws for Routed Language Models", ICML 2022, arXiv:2202.01169, App B.1-B.2.1** — S-BASE. Eq 17 = our entropy-regularised argmax; Eq 19 = our capacity constraint; Eq 23 = Sinkhorn as ALTERNATING DUAL ASCENT with `g in R^E`, ONE SCALAR PER EXPERT = our mu; calls it "the regularized Kantorovich problem" and "adding a balancing distribution constraint to the softmax operator". Ancestors: **Lewis et al., BASE Layers, ICML 2021, arXiv:2103.16716**; **Zhou et al., Expert Choice, NeurIPS 2022, arXiv:2202.09368** (entropy-regularised LP, Dykstra, claims no aux loss) |

ALSO: **Liao, "Transformer as an Euler Discretization of Score-based Variational Flow",
arXiv:2604.23740 (Apr 2026)** — single-author unpublished preprint whose abstract is our thesis:
MHA "approximates SVFlow vector field via a vMF kernel-smoothed posterior", MoE/FFN "in a relaxed
network-based way", and it claims to explain "why attention trains stably without explicit
regularization while MoE requires auxiliary balancing losses". Preprint status is our only protection.

WHAT SURVIVES AS NOVEL (searched for, not found): (i) identifying the log-sum-exp Hopfield
gradient as an **MoE with per-expert MLPs** — Hyper-SET's `d_m` are single vectors, not experts,
and it derives no router; (ii) the **"chemical potential" / grand-canonical change-of-ensemble**
framing — zero prior hits, but it is terminology over Clark et al.'s mathematics.

---

## 2026-09-22 — INTERNAL-CONSISTENCY AUDIT after `tab:main` landed (self-caught)

Introducing the generated `tab:main` (32.0B tokens, `configs/cmix/` 70/30 web/math) put it in the
same paper as three main-text tables built from the **7.86B-token, 100%-web** `iclr_*` grid. Two of
those three stated no budget and no corpus, so a reader comparing across them is comparing two
different experiments.

**The specific trap.** `tab:pure`'s energy row is `Avg11 = 44.58`; `tab:main`'s pure row is
`42.00`. Same *word* ("pure"/energy backbone), 2.58pp apart, and nothing in either caption said
why. They are different arms on different corpora at different budgets:

| | `tab:pure`, `tab:threeway` | `tab:main`, `tab:cmix134m` |
|---|---|---|
| tokens | **7.86B** (30,000 steps x 262,144) | **32.0B** (122,070 x 262,144) |
| corpus | **100% web** (`web-nemotron-cc-hq-p2_{0,1}` only) | **70/30** (+`megamath-web-pro_0`, `finemath-3plus-rewritten_0`) |

Verified from the config, not from memory: `configs/iclr_sink/iclr_hop_K32_top2_sink.yml` has
`num_training_steps: 30000`, `micro_batch_size: 4`, `gradient_accumulation_steps: 4`, and a
`datasets:` block containing **only** the two web sets.

**Fixed** (`sec/experiments.tex`): `tab:pure` and `tab:threeway` captions now state the budget and
corpus in bold and say explicitly that they are not comparable to `tab:main`. `tab:cost` needed
nothing — it is MACs/token only, budget-independent. `tab:cmix134m` needed nothing — it agrees with
`tab:main` to the digit on all four shared rows (44.82/43.45/45.43 and FLOPwt 141.7/141.7/143.2).

**Also fixed: a dangling reference I introduced myself.** Moving the old `tab:frontier` into
`sec/outtakes.tex` left four live `\ref{tab:frontier-outtake}`. They all *resolve* (outtakes is
`\input`), so LaTeX is silent — but two were in the MAIN text, resting a claim on a table the user
intends to delete. `sec/experiments.tex:164` was simply mis-pointed: every number in its sentence
(44.83 energy+Switch, 44.58 energy+Boltzmann, 44.43 plain-stack+Switch) is a row of **`tab:pure`**,
and the plain-stack row is not in the frontier table at all. Repointed to `tab:pure`.

**STILL OPEN — needs a decision.** Three refs genuinely need the frontier table's content and have
no other source:
- `sec/experiments.tex:222` — the aux-loss ablation, `43.83 -> 43.12` (its last two rows).
- `sec/appendix.tex:1602` — the MMLU 24.2-26.6% range (its MMLU column).
- `sec/appendix.tex:1990` — the corrected-routing-sign statement.

`sec/outtakes.tex` already concedes the table "remains usable as a K/k sparsity sweep at 7.86B".
So it is **not** "too old" in the sense the user's instruction meant, and it should be promoted out
of outtakes into `sec/appendix.tex` as the sparsity-sweep table, keeping the `tab:frontier-outtake`
label so all four refs keep resolving. One-block move; not done unilaterally because it reverses a
placement the user asked for.

**A ranking in `tab:pure` that the loss does not support.** Final `train-lm_loss` against Avg11:

| row | Avg11 | final `lm_loss` |
|---|---|---|
| energy backbone + Switch SwiGLU (`iclr_switch_K16_top2`) | 44.83 | 3.1589 |
| energy backbone + Boltzmann Hopfield K=32 (`iclr_hop_K32_top2_sink`) | 44.58 | 3.1679 |
| plain GPT stack + Switch SwiGLU (`gptmoe_last_isoP`) | 44.43 | 3.1779 |
| plain GPT stack + Switch SwiGLU, 3x budget (`gptmoe_last_3x`) | 44.20 | **3.1387 (best)** |

The **best-loss arm scores worst on Avg11**. Across the three iso-param arms loss and Avg11 do
co-order, but over a span of 0.019 nats -- which at the paper's own ~4.5pp/nat slope predicts
0.086pp, against 0.40pp observed. The prose already claims parity rather than advantage and calls
the two contributions "the same size within noise", so no prose change was needed; the bolded
`44.83` is a within-table maximum, not an asserted win. Flagging it because the 3x row's inversion
is a clean argument that Avg11 does not track loss across param counts, and a reviewer may raise it.

**Method note (cost me a wrong number mid-audit).** `compute_avg11.py <run_dir>` globs
**recursively** and will silently pick up an ablation subdirectory. For
`iclr_hop_K32_top2_sink` it read `ablate/C_proxysel/harness_results_*.json` and returned
**44.55**, which I briefly treated as a drift from the paper's 44.58. Pointing it at
`<run_dir>/unsharded` returns **44.58** and confirms the paper. **Always pass the eval dir, not
the run dir, for any arm that has an `ablate/` subtree.**

## 2026-09-22 (b) — the loss column dissolves most of the 134M ranking

Added `lm_loss` to `tab:main` and checked every gap against the project's credibility rule
(`~2x of 4.52*dloss`). At 134M the five non-pure arms lie inside **0.0338 nats**, which predicts
0.15pp of Avg11 against 2.10pp observed — **13.7x** — and every pairing against the hybrid fails the
rule (6.7x, 6.3x, 57x, plus a **sign inversion** where dense iso-active has worse loss and better
Avg11). Only `pure` passes, worse by >=0.32 nats. At 400M the loss separates properly (0.0653 and
0.0704 nats) and the energy deficit in loss **triples** from 134M — the defensible form of "widens
with scale". One prior reading is **retracted**: recurrence changes the energy arm's loss by +0.0002
nats, so it improves nothing, and "repeated descent on an energy" is out of the caption.

A reviewer will ask why an 0.024-nat difference moves Avg11 by 0.73pp. We do not know; the honest
answer in the caption is that the ordering is reproducible but is not a language-modelling effect.

## 2026-09-22 (c) — TWO ABSTRACT CLAIMS THE TABLES NO LONGER SUPPORT (not edited; needs the user)

The abstract is the only place these appear; `sec/intro.tex` does not mention 1B at all.

**(1) "hybrid GPT+Boltzmann-MoE models show performs almost on par with standard switch-MoE."**
This now contradicts our own experiments section, which says "negative for energy routing at 400M".
Energy vs FLOP-matched Switch:

| tier | energy `lm_loss` | Switch `lm_loss` | dloss | Avg11 gap |
|---|---|---|---|---|
| 134M `6G1x6E` vs `6G1x6S` | 2.6544 | 2.6329 | +0.0215 | -0.61pp |
| 400M `6G1x6E` vs `6G1x6S` | 2.4748 | 2.4095 | +0.0653 | -0.95pp |
| 400M `6G6E` vs `6G6S` | 2.4746 | 2.4042 | +0.0704 | -1.91pp |

"Almost on par" is defensible at 134M and NOT at 400M, where the loss deficit **triples** and the
Avg11 gap reaches 1.91pp. An abstract that says "on par" against an experiments section that says
"negative" is the first thing a reviewer will notice.

*Proposed:* "We show that hybrid GPT+Boltzmann-MoE models are competitive with a standard
Switch-MoE at 134M parameters --- within 0.022 nats of training loss --- but that a FLOP-matched
learned gate pulls ahead at 400M, where the gap in loss triples."

**(2) "The sparse activation of Boltzmann-MoE makes scaling of recurrent blocks to larger models
feasible and we demonstrate this up to 1B scale hybrid models."**
**There is no recurrent 1B arm.** The only completed 1B model is `cmix1B_12L_gptDense_32B`:
1002.07M total / 279.32M active, `layer_iterations = [1]x12`, i.e. **no recurrence**, and its
structure is `8G4E` (8 softmax+MLP then 4 energy+BoltzmannMoE, K=64), a STACKED model, not the
`6G1x6E` hybrid. Verified from the config, not from the name.

Note the trap that nearly misled this check: `iclr_pure_hop_K16_top2_1blk*` and `slope90k_1blk*`
match a case-insensitive `*1b*` glob but are **one-BLOCK** recurrent arms at **104.60M**, not 1B
models. Only `cmix1B_*` is 1B.

*Proposed:* "The sparse activation of Boltzmann-MoE makes larger models feasible, and we
demonstrate a 1B-parameter model (279M active per token) in which four energy blocks with
Boltzmann-MoE feedforwards are stacked on eight standard transformer blocks." Then either drop
"recurrent" from the scaling claim or support it: our recurrent arms top out at 400M.

## 2026-09-22 (d) — TWO CODE BUGS FROM THE USER'S AUDIT. Both confirmed, both touch published arms.

### A. `proxy_init: svd` never ran. Silent no-op on nine arms, including the 134M baseline.

`_svd_refit_proxy()` writes per-expert with `self.proxy_V.data[k].copy_(plain_tensor)`. Under FSDP
`proxy_V`/`proxy_B` are sharded DTensors and the SVD factors are plain, so the first write raises:

```
RuntimeError('aten.copy_.default got mixed torch.Tensor and DTensor, ...')
```

A bare `except Exception` logs a warning and sets `_svd_done = True`, so it never retries. **893
occurrences across 36 logs.** Affected (all specify `svd`, all actually got RANDOM init):
`cmix_134M_hybrid_32B_sparse` (BASELINE), `cmix_134M_sandwich_32B_sparse`,
`cmix_134M_pure_32B_sparse`, `cmix_400M_hybrid_sparse`, `cmix_400M_sandwich_sparse`, `abl_C`,
`abl_R`, `abl_G_400M_6G1x6E1x6E`, `abl_H_400M_6G6E_deep`.

The function's own docstring measures top-2 router agreement on a trained 1B checkpoint at **0.5541
SVD vs 0.0378 random** -- so a ~15x better starting point was specified and silently discarded
everywhere.

**The audit's stated symptom is not the one that bites.** "Re-initialised from SVD at every resume"
is a correct reading of the guard (`_svd_done` is a plain attribute, not a buffer, so it resets per
process, and a resume past `sparse_start_step` does re-trigger the transition). But the refit dies on
its first write, so nothing is overwritten. **Fixing the DTensor bug alone would have CREATED the
clobbering bug.** Both were fixed together: full-tensor writes through a new `_copy_full_into`
(redistributes when the target is sharded), plus a persistent `_svd_done_buf` buffer.

### B. Every auxiliary loss is multiplied by 0.001, and the Switch baselines use 10x more.

`model_wrapper/pretraining.py:543` -> `lm_loss + router_aux_loss_coef * aux_loss`, default **0.001**
(`CommonConfig`), **unset in every energy config**. Effective weights: `repulsion_coef: 0.1` -> 1e-4;
`proxy_loss_coef: 0.01` -> 1e-5.

**The asymmetry:** the only two configs that set the field are
`configs/iclr_moebase/iclr_switch_K16_top2{,_shared}.yml`, both at **0.01 = 10x** the default the
energy arms use. The Switch baseline's load-balance loss therefore carries ten times the weight of the
energy arms' auxiliary losses, in arms compared directly against each other.

**This likely invalidates §19.5's "no knob improves Boltzmann MoE".** Repulsion at an effective 1e-4
would do almost nothing regardless of its nominal value, so that sweep may have measured a disabled
mechanism rather than an ineffective one. **Do not quote §19.5 until the sweep is re-run.**

### Handling
`abl_S` and `abl_P` were killed and will be relaunched on fixed code. `abl_C` (98% complete) was left
to finish under the old code, so it stays consistent with the published set. Fix A is validated by a
400-step 2-GPU run that crosses `sparse_start_step`; the pass criterion is zero "proxy SVD refit
skipped" lines.

## 2026-09-22 (e) — the sparsity claim had never been measured

**Severity: highest so far.** Not a wrong number but a missing measurement, in the one place the
paper's contribution lives.

`_sparse_active` is set in `__init__` from `sparse_start_step` and flipped only by
`set_training_step()`, which only the training loop calls. Every checkpoint trains with
`sparse_start_step > 0` (the dense warmup distils the proxy), so **every eval we have published ran
the dense all-K path with exact oracle routing.** The proxy / surrogate router — the mechanism that
makes the sparsity real, and the thing the reviewers will ask about — never ran at eval time.

What a reviewer would do with this: "you report a sparse model's accuracy but you evaluate it
densely, so your accuracy is an upper bound you cannot deliver at your claimed FLOPs." That is a
reject, and it would be correct.

**What must change in the draft once the wave-A sparse evals land:**

1. Every sparse arm needs **both** columns — oracle and proxy-sparse — or the sparse one alone. A
   single unlabelled accuracy for a sparse arm is exactly the ambiguity that caused this.
2. `app:eval` must state which routing path produced each number. It currently does not, because
   until today there was only one path in practice.
3. The FLOPs/MACs story must be consistent with the *evaluated* path. Quoting sparse MACs beside
   dense-oracle accuracy is the composite claim that cannot be defended.
4. §21.4: these routers were distilled at an effective `1e-5` weight because of the
   `router_aux_loss_coef: 0.001` multiplier (Bug B). If the sparse penalty turns out small even so,
   that is a *strong* result and worth saying plainly — the proxy works despite being barely trained.
   If it turns out large, Bug B is the leading explanation and the honest move is to say the sparse
   configuration is not yet validated rather than to bury it.

**Process lesson.** `auto_eval_on_finish.sh:101` documented the symptom in a comment and worked
around the OOM it caused. A comment that explains why a workaround is needed is a comment describing
a bug; it should have been escalated, not accommodated. Worth grepping the tree for other
"reloads with"/"falls back to" comments that encode a silent behavioural difference.

## 2026-09-22 (f) — `tab:cost`'s "proxy router" row mixes two different models

Distinct from (e) and in the MAIN paper, not the appendix.

`tab:cost` row "Boltzmann, $K{=}32$, proxy router" reports Avg11 **44.58**. That number comes from
`iclr_hop_K32_top2_sink/unsharded/`, an export with `sparse_forward` absent and `proxy_rank: 0` —
i.e. **no proxy at all**. The arm's real proxy export (`unsharded_sparse_r16m512`, r=16, p=2 of 32)
had never been evaluated. So the row's cost columns describe the proxy model and its accuracy column
describes the dense one.

A reviewer asking "what does the proxy cost you in accuracy?" would find the table answers
"nothing", because the accuracy shown is the accuracy without it. Two further arms
(`iclr_big_hop_sandwich_sink`, `pure_hop_T12_sink` — the source of the 40.96 pure number) have the
same unevaluated-sparse-export pattern.

**Required:** no row may combine a cost measured on one export with an accuracy measured on another.
Either both from the sparse export, or the row states explicitly that accuracy is oracle-routed and
gives the proxy's accuracy separately. Evals for all three sparse exports are running.

**Pattern to watch for generally:** an arm directory with several `unsharded*` exports differing in
configuration is a trap, because the eval tooling globs for the newest `harness_results*.json` and
nothing ties a number back to which export produced it. The `sparseeval_*` convention introduced
today at least makes the routing path visible in the path name.
