#!/bin/bash
# TIER 2: the two web .bin files, 2,000 GB, onto the dmfexp fileset.
# STAGED BUT NOT SUBMITTED -- needs a decision, because dmfexp is shared with colleagues and
# holds our live training checkpoints.
#
# Only worth running if the whole `datasets` FILESET is at risk. If the owners are merely deleting
# their own files, the hard links already in /proj/datasets/ndehmamy-dataset-rescue/ keep the data
# alive at zero cost and this copy adds nothing.
#
# Skips the .bytes files (1,005 GB) deliberately: only byte_megatron_dataset.py reads them and our
# configs are MegatronDataset.
set -uo pipefail
SRC=/proj/datasets/granite-4-datasets-megatron-merged
DST=/proj/dmfexp/nima/datasets-rescue
MIN_FREE_GB=${MIN_FREE_GB:-800}          # refuse to drive dmfexp below this
mkdir -p "$DST"
rc=0
for f in web-nemotron-cc-hq-p2_0.bin web-nemotron-cc-hq-p2_1.bin; do
  s="$SRC/$f"; d="$DST/$f"
  [ -f "$s" ] || { echo "MISSING SOURCE $f"; rc=1; continue; }
  ssz=$(stat -c %s "$s")
  if [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$ssz" ]; then echo "already present: $f"; continue; fi
  need=$(( ssz / 1024 / 1024 / 1024 ))
  free=$(df -BG "$DST" | awk 'NR==2{gsub("G","",$4); print $4}')
  echo "[$(date -u +%FT%TZ)] $f needs ${need}G; dmfexp free ${free}G; floor ${MIN_FREE_GB}G"
  if [ $(( free - need )) -lt "$MIN_FREE_GB" ]; then
    echo "REFUSING $f: would leave $(( free - need ))G, under the ${MIN_FREE_GB}G floor." >&2
    echo "Raise the dmfexp fileset quota, or lower MIN_FREE_GB deliberately." >&2
    rc=1; continue
  fi
  cp --sparse=always "$s" "$d.part" || { echo "COPY FAILED $f"; rm -f "$d.part"; rc=1; continue; }
  if [ "$(stat -c %s "$d.part")" != "$ssz" ]; then echo "SIZE MISMATCH $f"; rm -f "$d.part"; rc=1; continue; fi
  mv -f "$d.part" "$d"; echo "[$(date -u +%FT%TZ)] ok $f"
done
echo "[$(date -u +%FT%TZ)] tier2 rc=$rc; dmfexp avail $(df -BG "$DST" | awk 'NR==2{print $4}')"
exit $rc
