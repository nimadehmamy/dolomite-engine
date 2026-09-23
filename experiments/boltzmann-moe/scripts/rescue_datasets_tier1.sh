#!/bin/bash
# TIER 1 dataset rescue: copy the SMALL, high-value files to a DIFFERENT fileset (dmfexp),
# so they survive even if the whole `datasets` fileset is lost -- which hard links cannot protect
# against.
#
# What and why:
#   the two math .bin+.idx (128.7 GB) -- they are what makes our 70/30 datamix distinctive, and
#                                       they are small enough to copy without endangering dmfexp
#   all four .idx (14.7 GB)          -- a .bin is useless without its .idx, and the .idx encodes
#                                       the exact document boundaries of this tokenisation
# NOT copied: the two web .bin (2,000 GB, needs a decision -- dmfexp has only 3.7 TB free and
#             oscillates by +-1 TB) and the .bytes files (1,005 GB, only used by
#             byte_megatron_dataset.py; our configs are MegatronDataset).
set -uo pipefail
SRC=/proj/datasets/granite-4-datasets-megatron-merged
DST=/proj/dmfexp/nima/datasets-rescue
mkdir -p "$DST"
FILES="megamath-web-pro_0.bin megamath-web-pro_0.idx megamath-web-pro_0.ndocs
finemath-3plus-rewritten_0.bin finemath-3plus-rewritten_0.idx finemath-3plus-rewritten_0.ndocs
web-nemotron-cc-hq-p2_0.idx web-nemotron-cc-hq-p2_0.ndocs
web-nemotron-cc-hq-p2_1.idx web-nemotron-cc-hq-p2_1.ndocs"
echo "[$(date -u +%FT%TZ)] tier1 start; dmfexp avail before: $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"
rc=0
for f in $FILES; do
  [ -z "$f" ] && continue
  s="$SRC/$f"; d="$DST/$f"
  [ -f "$s" ] || { echo "  MISSING SOURCE $f"; rc=1; continue; }
  ssz=$(stat -c %s "$s")
  if [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$ssz" ]; then echo "  already present, size OK: $f"; continue; fi
  echo "  copying $f ($(numfmt --to=iec $ssz))"
  cp --sparse=always "$s" "$d.part" || { echo "  COPY FAILED $f"; rc=1; rm -f "$d.part"; continue; }
  dsz=$(stat -c %s "$d.part")
  if [ "$dsz" != "$ssz" ]; then echo "  SIZE MISMATCH $f: src=$ssz dst=$dsz"; rc=1; rm -f "$d.part"; continue; fi
  mv -f "$d.part" "$d"
  echo "    ok $f"
done
echo "[$(date -u +%FT%TZ)] tier1 done rc=$rc; dmfexp avail after: $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"
echo "--- verifying sizes against source ---"
for f in $FILES; do
  [ -z "$f" ] && continue
  s="$SRC/$f"; d="$DST/$f"
  [ -f "$d" ] || { echo "  MISSING COPY $f"; rc=1; continue; }
  [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ] && echo "  OK   $f" || { echo "  BAD  $f"; rc=1; }
done
exit $rc
