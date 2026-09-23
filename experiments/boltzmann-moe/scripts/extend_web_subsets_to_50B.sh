#!/bin/bash
# Extend the shared web subsets from 20B to 50B tokens each, so 128B-token runs are possible.
#
# 128B at weight 0.35 with split 99,0.5,0.5 needs 45.3B train tokens per shard. 50B gives 1.10x
# headroom; the bare 45.3B would risk the blend sampler wrapping into a silent second epoch.
#
# Built to *.new.{bin,idx} and swapped in only after verification, so a failure or a preemption
# leaves the existing working 20B data untouched.
#
# !! INVALIDATES THE BLEND CACHE !! /proj/dmfexp/datasets-shared/.cache indexes the 20B documents.
# Leaving it would either error or silently sample the old document set, so it is removed and the
# smoke test rebuilds it.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
SRC=/proj/datasets/granite-4-datasets-megatron-merged
S=/proj/dmfexp/datasets-shared/granite-4-cmix-subset
TOK=${TOK:-50e9}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null || true
rc=0
echo "[$(date -u +%FT%TZ)] extend to $TOK tokens/shard; dmfexp avail $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"

for b in web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1; do
  [ -f "$SRC/$b.bin" ] || { echo "SOURCE GONE: $b.bin -- cannot extend"; rc=1; continue; }
  have=$(python3 - "$S/$b.idx" <<'PY'
import sys,numpy as np
sys.path.insert(0,'/proj/dmfexp/nima/Code/dolomite-engine')
from lm_engine.data.megatron.indexed_dataset import _IndexReader
print(int(_IndexReader(sys.argv[1],multimodal=False).sequence_lengths.sum(dtype=np.int64)))
PY
)
  echo "  $b currently holds ${have} tokens"
  python3 "$EXP/scripts/build_dataset_subset.py" --src "$SRC/$b" --dst "$S/$b.new" --tokens "$TOK" \
    2>&1 | grep -vE "INFO|WARNING|^W[0-9]" || { rc=1; continue; }
  # verify the NEW file against the source before swapping
  python3 - "$SRC/$b" "$S/$b.new" <<'PY' || { echo "  VERIFY FAILED -- not swapping"; rc=1; continue; }
import sys,os,numpy as np
sys.path.insert(0,'/proj/dmfexp/nima/Code/dolomite-engine')
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset,_IndexReader,DType
src,dst=sys.argv[1],sys.argv[2]
rs=_IndexReader(src+'.idx',multimodal=False); rd=_IndexReader(dst+'.idx',multimodal=False)
Ls,Ld=rs.sequence_lengths,rd.sequence_lengths; S=len(Ld); it=DType.size(rd.dtype)
assert np.array_equal(Ld,Ls[:S]), "lengths not a prefix of the source"
assert int(Ld.sum(dtype=np.int64))*it==os.path.getsize(dst+'.bin'), "idx tokens != bin bytes"
a=MMapIndexedDataset(src,multimodal=False); c=MMapIndexedDataset(dst,multimodal=False)
for i in [0,1,S//3,S//2,S-2,S-1]:
    assert np.array_equal(np.asarray(a[i]),np.asarray(c[i])), f"doc {i} differs"
print(f"  verified: {S:,} seqs, {int(Ld.sum(dtype=np.int64)):,} tokens, 6/6 docs byte-identical")
PY
  mv -f "$S/$b.new.bin" "$S/$b.bin"
  mv -f "$S/$b.new.idx" "$S/$b.idx"
  [ -f "$S/$b.new.ndocs" ] && mv -f "$S/$b.new.ndocs" "$S/$b.ndocs"
  chgrp proj_dmfexp "$S/$b.bin" "$S/$b.idx" "$S/$b.ndocs" 2>/dev/null
  chmod 0664 "$S/$b.bin" "$S/$b.idx" "$S/$b.ndocs" 2>/dev/null
  echo "  swapped in: $b"
done

if [ $rc -eq 0 ]; then
  echo "  removing the stale blend cache (it indexes the old 20B document set)"
  rm -rf /proj/dmfexp/datasets-shared/.cache
fi
echo "[$(date -u +%FT%TZ)] rc=$rc; shared total $(du -sh /proj/dmfexp/datasets-shared 2>/dev/null|cut -f1); dmfexp avail $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"
exit $rc
