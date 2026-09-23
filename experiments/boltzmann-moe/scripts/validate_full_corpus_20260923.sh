#!/bin/bash
# Validate the full-corpus copy in /proj/dmfexp/datasets-shared/granite-4-cmix-FULL.
#
# A size match alone is NOT enough. The failure this guards against is a .bin/.idx pairing that is
# individually well-formed but mutually wrong: full .idx + subset .bin reads past EOF, subset .idx +
# full .bin silently trains on the first 18.6%, and NEITHER errors. So the decisive check is the
# token count the .idx reports against the .bin's actual length.
set -uo pipefail
SRC=/proj/datasets/ndehmamy-dataset-rescue
DST=/proj/dmfexp/datasets-shared/granite-4-cmix-FULL
SETS="web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1 megamath-web-pro_0 finemath-3plus-rewritten_0"
fail=0
echo "=== 1. byte sizes, src vs dst ==="
for f in $SETS; do for e in bin idx ndocs; do
  a=$(stat -c %s "$SRC/$f.$e" 2>/dev/null); b=$(stat -c %s "$DST/$f.$e" 2>/dev/null)
  if [ "$a" = "$b" ] && [ -n "$a" ]; then printf "  OK       %-34s %-4s %s\n" "$f" "$e" "$a"
  else printf "  MISMATCH %-34s %-4s src=%s dst=%s\n" "$f" "$e" "${a:-NA}" "${b:-NA}"; fail=1; fi
done; done
echo "=== 2. no leftover .part files (an interrupted copy) ==="
p=$(ls "$DST"/*.part 2>/dev/null | wc -l); echo "  .part files: $p"; [ "$p" -eq 0 ] || fail=1
echo "=== 3. content spot-check: first/middle/last 8 MiB of each .bin ==="
for f in $SETS; do
  sz=$(stat -c %s "$DST/$f.bin"); mid=$(( sz/2/8388608*8388608 )); last=$(( (sz-8388608)/8388608*8388608 ))
  for off in 0 $mid $last; do
    if cmp -s -i ${off}:${off} -n 8388608 "$SRC/$f.bin" "$DST/$f.bin"; then :
    else echo "  CONTENT DIFF $f.bin at offset $off"; fail=1; fi
  done
  echo "  spot-checked $f.bin at 0 / $mid / $last"
done
echo "=== 4. THE DECISIVE CHECK: token count the .idx reports, and .bin consistency ==="
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
python - "$DST" "$SRC" <<'PY'
import sys, os
sys.path.insert(0,"/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.data.megatron.indexed_dataset import _IndexReader, MMapIndexedDataset
dst, src = sys.argv[1], sys.argv[2]
bad = 0
for f in ("web-nemotron-cc-hq-p2_0","web-nemotron-cc-hq-p2_1","megamath-web-pro_0","finemath-3plus-rewritten_0"):
    try:
        rd = _IndexReader(f"{dst}/{f}.idx", multimodal=False)
        n = int(rd.sequence_lengths.sum(dtype="int64")); docs = len(rd.sequence_lengths)
        rs = _IndexReader(f"{src}/{f}.idx", multimodal=False)
        ns = int(rs.sequence_lengths.sum(dtype="int64"))
        binsz = os.path.getsize(f"{dst}/{f}.bin")
        bpt = binsz / n
        ds = MMapIndexedDataset(f"{dst}/{f}")
        d0, dl = ds[0], ds[len(ds)-1]     # first and last document must both be readable
        ok = (n == ns) and 1.5 < bpt < 4.5 and len(d0) > 0 and len(dl) > 0
        print(f"  {'OK  ' if ok else 'BAD '} {f:34s} tokens={n:,} (src {ns:,}) docs={docs:,} "
              f"bytes/token={bpt:.2f} first_doc={len(d0)} last_doc={len(dl)}")
        if not ok: bad += 1
    except Exception as e:
        print(f"  BAD  {f:34s} {type(e).__name__}: {e}"); bad += 1
print(f"  token-count/readability failures: {bad}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=1
echo "=== 5. tokenizer present ==="
ls "$DST/tokenizers" >/dev/null 2>&1 && echo "  OK tokenizers/" || { echo "  MISSING tokenizers/"; fail=1; }
echo
[ "$fail" -eq 0 ] && echo "VALIDATION PASSED -- safe to point training at $DST" || echo "VALIDATION FAILED -- do NOT train on this copy yet"
exit $fail
