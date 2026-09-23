#!/bin/bash
# Unshard a checkpoint into <out_dir> so that the result is NEVER a partial file.
#
# HISTORY, because two earlier designs both failed and the failures looked like successes.
#
# v1 used `flock` on a lock file in /tmp. /tmp is NODE-LOCAL, so it serialised nothing whenever
# LSF placed the two eval jobs (ev_* likelihood, evg_* gsm8k -- submitted together by
# auto_eval_on_finish.sh) on different hosts, which is the normal case.
#
# v2 moved the flock file onto shared GPFS. THAT DID NOT WORK EITHER, and it was verified the
# wrong way: an idempotency test only exercises the `-f model.safetensors` short-circuit, not the
# lock. Measured on abl_H_400M_6G6G_deep_isoactive (jobs 1856852 on p6-r08-n2 and 1856853 on
# p6-r28-n1): both logged "loading checkpoint" at 13:48:18.449 and 13:48:19.062, i.e. 613 ms
# apart, so both ran lm_engine.unshard into the same directory. flock does not give cross-node
# mutual exclusion here.
#
# v3 (this one) DOES NOT RELY ON LOCKING FOR CORRECTNESS. Each caller unshards into its own
# private temp directory and then moves each file into place with `mv -n`, which within one
# filesystem is rename(2) and therefore atomic: $OUT/model.safetensors either does not exist or
# is a complete file. Two concurrent callers duplicate ~20 s of work, which is cheap; what they
# can no longer do is interleave writes into the same file.
#
# Both earlier outputs happened to be correct, because unsharding is deterministic and the two
# processes wrote identical bytes at identical offsets. That is luck, not a guarantee -- a
# truncation by one after the other had advanced would leave a hole.
set -u
LOAD_PATH=$1; ITER=$2; OUT=$3
REPO=/proj/dmfexp/nima/Code/dolomite-engine

if [ -f "$OUT/model.safetensors" ]; then
    echo "unshard: already present at $OUT"
    exit 0
fi

TMP="${OUT}.tmp.$(hostname -s).$$"
rm -rf "$TMP"; mkdir -p "$TMP" || exit 1
trap 'rm -rf "$TMP"' EXIT

C=$(mktemp /tmp/unsh_XXXXXX.yml)
printf 'load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n' \
    "$LOAD_PATH" "$ITER" "$TMP" > "$C"
cd "$REPO" && python -m lm_engine.unshard --config "$C"
rc=$?
rm -f "$C"
[ $rc -ne 0 ] && { echo "unshard: FAILED rc=$rc" >&2; exit $rc; }

if [ -f "$OUT/model.safetensors" ]; then
    echo "unshard: another writer finished first; discarding our copy"
    exit 0
fi

mkdir -p "$OUT" || exit 1
# `mv -n` is rename(2) within a filesystem: atomic, and never clobbers a file another writer
# already put there. Move model.safetensors LAST so the directory is only ever advertised as
# complete once everything beside it is in place ( `-f model.safetensors` is the readiness flag).
for f in "$TMP"/*; do
    b=$(basename "$f")
    [ "$b" = "model.safetensors" ] && continue
    mv -n "$f" "$OUT/$b" 2>/dev/null
done
if [ -f "$TMP/model.safetensors" ]; then
    mv -n "$TMP/model.safetensors" "$OUT/model.safetensors" 2>/dev/null
fi
echo "unshard: wrote $OUT"
exit 0
