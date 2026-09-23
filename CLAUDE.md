# dolomite-engine — project-level instructions

## Boltzmann-MoE ICLR 2026 paper — Overleaf location

**The active paper draft lives outside this repo, in an Overleaf git-bridge clone:**
`~/Code/overleaf/boltzmann-moe-ICLR-2026/` (remote `origin`, branch `main` — pushing there
IS how the user sees the change, since that push goes straight to Overleaf). Bib file:
`boltz_moe.bib` (only bib file actually `\bibliography`'d by `main.tex`).

Full editing workflow, style rules, and paper structure are documented in
[`experiments/boltzmann-moe/CLAUDE.md`](experiments/boltzmann-moe/CLAUDE.md) — read that
file before touching the draft. It covers: the "edit directly" pull/edit/push round trip,
the sign-convention and no-em-dash style rules, the 8-9 page main-body limit, and where
each table (`tab:frontier`, `tab:pure`, `tab:cost`, `tab:threeway`, etc.) lives in
`sec/experiments.tex` / `sec/appendix.tex`.

## Register-decode bug fix (2026-06-29)

**Generation outputs from all `register_energy` checkpoints (V0, V1, V73, V41,
V56, V76, h1, all Hopfield variants — i.e. any `model_type: register_energy`)
are corrupted by an off-by-R RoPE position bug in the cached-decode path.**

- Likelihood-based evals (avg10_norm, perplexity, log-prob multiple-choice)
  are **unaffected** — they run a single prefill forward.
- Generation-based evals (`.generate()`, BBH free-form, GSM8K) are
  **catastrophically degraded**. Empirical example: hop_r256 step 24000 →
  BBH=0.0738 (random ~0.23), GSM8K=0.0%, while avg10_norm=0.5112 (=base).
- The bug was: `config.register_generation_mode` was documented in comments
  but never read by the code; only `register_start_layer` was. Patching
  `bypass` → `no_cache` on a config.json had no effect.

**Fix applied:** `register_generation_mode` is now wired up in both
`register_energy/main.py:prepare_inputs_for_generation` and
`register_energy/model.py:forward`. Default kept as `"bypass"` for backward
compat. Switch any checkpoint to `"no_cache"` by editing its `config.json` —
no retraining needed.

**Action item for future agents:** every published register-equipped
BBH/GSM8K number in this repo predates this fix. Re-run with
`register_generation_mode: "no_cache"` before drawing any conclusions about
whether registers help or hurt generation.

Full writeup:
[`lm_engine/hf_models/models/register_energy/REGISTER_DECODE_BUG.md`](lm_engine/hf_models/models/register_energy/REGISTER_DECODE_BUG.md).
