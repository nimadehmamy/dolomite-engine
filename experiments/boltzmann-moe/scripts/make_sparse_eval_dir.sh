#!/bin/bash
# Create a parallel eval directory whose config.json has sparse_start_step: 0, so the model
# reloads with _sparse_active TRUE and the eval runs the PROXY-SPARSE forward path.
#
# WHY (2026-09-22). `_sparse_active = bool(sparse_forward) and sparse_start_step <= 0` is set in
# __init__ and only ever flipped by set_training_step(), which is called ONLY from the training loop
# (pretrain.py:403). So every checkpoint trained with sparse_start_step > 0 was EVALUATED on the
# dense all-K path with EXACT (oracle) routing -- never through the proxy router whose whole purpose
# is to make the sparsity real. auto_eval_on_finish.sh already documents the symptom ("reloads with
# _sparse_active FALSE, so eval runs the DENSE all-K path and single-block arms OOM at batch 4") but
# worked around the OOM rather than the cause.
#
# The author's own precedent for how much eval-time router treatment matters: dropping mu at eval
# cost 5.33pp Avg11 on a pure-energy arm (energy_ff.py:781-792).
#
# Weights are HARD LINKED, so this costs no space and cannot diverge from the original.
# The ORIGINAL directory is left untouched, so the oracle numbers remain for comparison.
set -uo pipefail
SRC=${1:?source unsharded dir}
DST=${2:?destination dir}
[ -f "$SRC/config.json" ] || { echo "no config.json in $SRC" >&2; exit 1; }
mkdir -p "$DST"
for f in "$SRC"/*; do
    b=$(basename "$f")
    case "$b" in
        harness_results*|gsm8k_raw*) continue ;;          # do not inherit the oracle results
        config.json) continue ;;                           # patched below
    esac
    [ -f "$f" ] && ln -f "$f" "$DST/$b" 2>/dev/null || cp -a "$f" "$DST/$b" 2>/dev/null
done
python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json,sys
src,dst=sys.argv[1],sys.argv[2]
c=json.load(open(src))
n=r=0
for b in (c.get('mlp_blocks') or []):
    if 'n_experts' not in b: continue
    if b.get('sparse_forward') and (b.get('sparse_start_step') or 0)>0:
        b['sparse_start_step']=0; n+=1
    # TRAINING-ONLY knob that blocks CONSTRUCTION at eval (found 2026-09-22 on the iclr_sink
    # sparse exports): energy_ff.py:945 asserts repulsion_subsample > 0 whenever sparse_forward
    # meets repulsion_space != "weight", because the sparse path never computes all K expert
    # outputs. But expert repulsion is an AUXILIARY LOSS gated on self.training -- it never
    # touches the forward output -- so it cannot arise at eval, and the comment at :952 says
    # exactly that ("Asserting at construction would make every such checkpoint unloadable for a
    # reason that cannot arise at eval"). This assert slipped past that intent. Zeroing the coef
    # is therefore EXACT for inference, and strictly safer than inventing a subsample size.
    if (b.get('sparse_forward') and float(b.get('repulsion_coef') or 0) > 0
            and int(b.get('repulsion_subsample') or 0) <= 0
            and b.get('repulsion_space') != 'weight'):
        b['repulsion_coef'] = 0.0; r += 1
json.dump(c,open(dst,'w'),indent=2)
print(f"  patched {n} block(s) to sparse_start_step: 0; neutralised training-only repulsion on {r}")
PY
echo "  sparse-eval dir ready: $DST"
