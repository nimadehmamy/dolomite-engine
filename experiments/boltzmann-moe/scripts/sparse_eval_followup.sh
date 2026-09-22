#!/bin/bash
# Backstop so the 2026-09-22 sparse-eval bug (HANDOFF §21) cannot silently recur on a NEW arm.
#
# Any checkpoint trained with sparse_forward + sparse_start_step > 0 reloads with _sparse_active
# FALSE and is evaluated on the DENSE all-K oracle path. auto_eval_on_finish.sh still produces that
# dense eval (deliberately -- it is the oracle baseline we compare against). This script notices any
# arm that has a dense eval but no PROXY-SPARSE eval and submits the sparse one.
#
# Written as a separate script rather than folded into auto_eval_on_finish.sh on purpose: that
# driver is what every future eval depends on, and with a deadline running it is not worth the risk
# of breaking it. This one is additive and defensive.
#
# SAFETY RAILS
#   * skips /milestones/ and unsharded_mucal* -- intermediate checkpoints are DEFERRED until every
#     final has been re-evaluated (user instruction 2026-09-22)
#   * requires an existing DENSE eval, so it only ever follows up work already deemed worth evaluating
#   * caps submissions per invocation (MAXSUB, default 4) so a scan cannot flood the queue
#   * runs the harness OFFLINE -- a burst of MMLU downloads got our IP HTTP-429'd
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
cd "$EXP"
MAXSUB=${MAXSUB:-4}
LIKE="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,mmlu,wikitext"
VENV="source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate && export PYTHONPATH=$REPO:\$PYTHONPATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export HF_HUB_OFFLINE=1 && export HF_DATASETS_OFFLINE=1"
LEDGER=$EXP/logs/sparse_eval_ledger.tsv

CAND=$(python3 - <<'PY'
import json,glob,os
out=[]
for cfg in glob.glob("results/**/unsharded*/config.json",recursive=True):
    d=os.path.dirname(cfg)
    if "/milestones/" in d or os.path.basename(d).startswith("unsharded_mucal"): continue
    try: c=json.load(open(cfg))
    except Exception: continue
    if not any(b.get('sparse_forward') and int(b.get('sparse_start_step') or 0)>0
               for b in (c.get('mlp_blocks') or [])): continue
    if not glob.glob(d+"/harness_results*.json"): continue          # no dense eval yet -> nothing to follow up
    # skip junk early checkpoints (step 500 / 4000 smoke saves): a sparse eval of an
    # undertrained proxy tells us nothing and costs a GPU-hour.
    b=os.path.basename(d)
    if b.startswith("unsharded_step"):
        try:
            if int(b.replace("unsharded_step","")) < 10000: continue
        except ValueError: pass
    tail=os.path.basename(d).replace("unsharded","sparseeval")
    dst=os.path.join(os.path.dirname(d),tail)
    if glob.glob(dst+"/harness_results*.json"): continue            # sparse eval already done
    out.append(f"{d}\t{dst}")
print("\n".join(out))
PY
)
[ -z "$CAND" ] && { echo "$(date -u +%FT%TZ) sparse_eval_followup: nothing to do"; exit 0; }

n=0
while IFS=$'\t' read -r src dst; do
  [ -z "${src:-}" ] && continue
  [ "$n" -ge "$MAXSUB" ] && { echo "  cap $MAXSUB reached; remaining arms picked up next cycle"; break; }
  # The STEP must be in the job name. Truncating to 24 chars made abl_R@122070 and abl_R@32000
  # collide, and the name check then skipped a second, legitimately different checkpoint.
  arm=$(echo "$src" | sed 's|results/||; s|/unsharded.*||; s|.*/||' | cut -c1-18)
  stp=$(basename "$src" | sed 's|^unsharded_*||')
  JN="sev_${arm}_${stp:-last}"
  bjobs -noheader -o job_name 2>/dev/null | grep -qx "$JN" && { echo "  skip $JN (queued)"; continue; }
  # Dedup on the DESTINATION DIR, not the job name: the wave fan-out spells its job names
  # differently (sev_sp134_pure vs sev_cmix_134M_pure_32B_sparse), so a name check alone would
  # happily submit a duplicate of a job already running against the same dir.
  BUSY=0
  while read -r pj; do
      [ -n "$pj" ] && [ "$pj" != NA ] || continue
      st=$(bjobs -noheader -o stat "$pj" 2>/dev/null)
      case "${st:-}" in PEND|RUN) BUSY=1 ;; esac
  done < <(awk -F'\t' -v D="$dst" '$5==D {print $2}' "$LEDGER" 2>/dev/null)
  [ "$BUSY" = 1 ] && { echo "  skip $dst (a job is already in flight against it)"; continue; }
  [ -f "$dst/config.json" ] || bash scripts/make_sparse_eval_dir.sh "$src" "$dst" >/dev/null || continue
  A=$EXP/$dst
  JID=$(bash scripts/bsub/submit_gpu_test.sh "$JN" \
"cd $REPO && $VENV && python experiments/eval_scripts/eval_harness.py --model hf \
--model_args pretrained=$A,dtype=bfloat16,trust_remote_code=True --tasks $LIKE --device cuda:0 \
--batch_size 1 --trust_remote_code --output_path $A/harness_results.json" \
 1 03:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
  echo "$(date -u +%FT%TZ) sparse_eval_followup: submitted $JN job=${JID:-NA} -> $dst"
  echo -e "$(date -u +%FT%TZ)\t${JID:-NA}\t$JN\tlike\t$dst\t$src" >> "$LEDGER"
  n=$((n+1)); sleep 4
done <<< "$CAND"
echo "$(date -u +%FT%TZ) sparse_eval_followup: $n submitted"
