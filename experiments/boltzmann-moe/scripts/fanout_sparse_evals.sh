#!/bin/bash
# Re-evaluate every sparse-trained checkpoint through the PROXY-SPARSE forward path.
#
# WHY (2026-09-22): `_sparse_active = bool(sparse_forward) and sparse_start_step <= 0` is set in
# __init__ and flipped ONLY by set_training_step(), called only from the training loop
# (pretrain.py:403). So every checkpoint trained with sparse_start_step > 0 was EVALUATED dense
# all-K with EXACT (oracle) routing -- the proxy/surrogate router never ran at eval, which is the
# one thing the sparsity claim rests on.
#
# FIX (validated, zero code change): a parallel eval dir with hard-linked weights whose config.json
# says sparse_start_step: 0. Probe on the 134M hybrid (job 1866475) moved all four tasks
# (copa -0.0100, sciq +0.0090, piqa -0.0011, winogrande -0.0071) with no traceback/OOM, proving the
# forward path changed. Originals are left untouched, so oracle-vs-sparse stays measurable.
#
# WAVES: A = Table 1 + Table 6 FINAL checkpoints (32B / step 61035 / 122070) -- the only thing that
#            matters for the paper.  B = short 8B-budget ablation finals (abl_T, abl_R@32k).
#        C = intermediate milestones: DEFERRED. Do NOT run until every wave-A final is redone
#            (user instruction, 2026-09-22).
# Usage: bash scripts/fanout_sparse_evals.sh [A|B|all] [--dry]
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
cd "$EXP"
LIKE="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,mmlu,wikitext"
GSM="gsm8k_cot"
VENV="source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate && export PYTHONPATH=$REPO:\$PYTHONPATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export HF_HUB_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && export HF_HUB_DISABLE_TELEMETRY=1 && export HF_DATASETS_TRUST_REMOTE_CODE=1"
WAVE=${1:-A}; DRY=${2:-}
LEDGER=$EXP/logs/sparse_eval_ledger.tsv
[ -f "$LEDGER" ] || echo -e "utc\tjobid\tjob_name\tkind\tsparse_eval_dir\toracle_dir" > "$LEDGER"

# tag<TAB>unsharded_dir<TAB>wave<TAB>want_gsm
read -r -d '' TARGETS <<'T'
sp134_hybrid	results/cmix32B/cmix_134M_hybrid_32B_sparse/unsharded_step122070	A	1
sp134_pure	results/cmix32B/cmix_134M_pure_32B_sparse/unsharded_step122070	A	1
sp134_sandwich	results/cmix32B/cmix_134M_sandwich_32B_sparse/unsharded_step122070	A	1
sp134_w1w2surr	results/cmix32B/cmix_134M_hyb_w1w2_sparse_surr_32B/unsharded_step122070	A	1
sp400_hybrid	results/cmix/cmix_400M_hybrid_sparse/unsharded_step61035	A	1
sp400_sandwich	results/tok32B/cmix_400M_sandwich_sparse/unsharded_step61035	A	1
sp400_ablH	results/iclr26_abl/abl_H_400M_6G6E_deep/unsharded_step61035	A	1
sp1B_gptDense	results/cmix/cmix1B_12L_gptDense_32B/unsharded_step61035	A	1
sp134_ablI	results/iclr26_abl/abl_I_134M_w1w2_sparse_surr_projUncon/unsharded_step122070	A	0
sp134_ablR	results/iclr26_abl/abl_R_134M_hyb_rnorm_none/unsharded_step122070	A	0
sp134_ablT035	results/iclr26_abl/abl_T_134M_hyb_tau0p35/unsharded_step32000	B	0
sp134_ablT2p0	results/iclr26_abl/abl_T_134M_hyb_tau2p0/unsharded_step32000	B	0
sp134_ablR32k	results/iclr26_abl/abl_R_134M_hyb_rnorm_none/unsharded_step32000	B	0
sp134_hyb_m8	results/cmix32B/cmix_134M_hybrid_32B_sparse/milestones/unsharded_tok8B_step32000	C	0
sp134_hyb_m16	results/cmix32B/cmix_134M_hybrid_32B_sparse/milestones/unsharded_tok16B_step62000	C	0
sp134_pure_m8	results/cmix32B/cmix_134M_pure_32B_sparse/milestones/unsharded_tok8B_step32000	C	0
sp134_sw_m8	results/cmix32B/cmix_134M_sandwich_32B_sparse/milestones/unsharded_tok8B_step32000	C	0
sp134_sw_m16	results/cmix32B/cmix_134M_sandwich_32B_sparse/milestones/unsharded_tok16B_step62000	C	0
sp134_sw_m24	results/cmix32B/cmix_134M_sandwich_32B_sparse/milestones/unsharded_tok24B_step92000	C	0
T

n=0
while IFS=$'\t' read -r tag src wv wgsm; do
  [ -z "${tag:-}" ] && continue
  [ "$WAVE" != "all" ] && [ "$wv" != "$WAVE" ] && continue
  [ -f "$src/config.json" ] || { echo "MISSING $src"; continue; }
  DST=$(dirname "$src")/sparseeval_$(basename "$src" | sed 's/^unsharded_*//')
  [ "$DST" = "$(dirname "$src")/sparseeval_" ] && DST=$(dirname "$src")/sparseeval
  if [ ! -f "$DST/config.json" ]; then
    bash scripts/make_sparse_eval_dir.sh "$src" "$DST" >/dev/null || { echo "FAILED to build $DST"; continue; }
  fi
  A=$EXP/$DST
  for kind in like gsm; do
    [ "$kind" = gsm ] && [ "$wgsm" != "1" ] && continue
    if [ "$kind" = like ]; then
      [ -f "$A/harness_results.json" ] && { echo "skip $tag like (done)"; continue; }
      JN="sev_$tag"; TASKS=$LIKE; OUT=$A/harness_results.json; W=03:00; POST="true"
    else
      G=$A/gsm8k; [ -f "$A/gsm8k_done" ] && { echo "skip $tag gsm (done)"; continue; }
      JN="sevg_$tag"; TASKS=$GSM; OUT=$G/harness_results.json; W=04:00
      POST="touch $A/gsm8k_done"
    fi
    bjobs -noheader -o job_name 2>/dev/null | grep -qx "$JN" && { echo "skip $JN (queued)"; continue; }
    CMD="cd $REPO && $VENV && python experiments/eval_scripts/eval_harness.py --model hf \
--model_args pretrained=$A,dtype=bfloat16,trust_remote_code=True --tasks $TASKS --device cuda:0 \
--batch_size 1 --trust_remote_code --output_path $OUT && $POST"
    if [ -n "$DRY" ]; then echo "DRY $JN -> $A"; n=$((n+1)); continue; fi
    JID=$(bash scripts/bsub/submit_gpu_test.sh "$JN" "$CMD" 1 $W 200G 2>&1 | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
    echo "submitted $JN job=$JID -> $A"
    echo -e "$(date -u +%FT%TZ)\t${JID:-NA}\t$JN\t$kind\t$DST\t$src" >> "$LEDGER"
    n=$((n+1))
    sleep 4          # 2026-09-22: stagger so a burst cannot re-trigger an HF 429
  done
done <<< "$TARGETS"
echo "=== $n job(s) for wave $WAVE ==="
