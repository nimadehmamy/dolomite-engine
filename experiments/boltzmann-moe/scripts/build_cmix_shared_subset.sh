#!/bin/bash
# Build a SHARED, space-efficient copy of the cmix datamix that can run new 32B-token jobs after
# /proj/datasets/granite-4-datasets-megatron-merged is deleted.
#
# Layout and sizing (per 32B run the blend draws 0.35/0.35/0.15/0.15):
#   web p2_0  20B tokens  74 GiB   <- 1.79x the 11.2B a 32B run draws; supports up to ~57B tokens
#   web p2_1  20B tokens  74 GiB
#   megamath  whole       48 GiB   <- only 13.0B tokens total; keeping it whole is simpler
#   finemath  whole       79 GiB   <- 21.1B tokens total
#   tokenizer             14 MiB
#   ~= 275 GiB, against ~4.4 TB free on the dmfexp fileset.
#
# The web shards are PREFIX SUBSETS built by build_dataset_subset.py, which is exact (see that
# file's header). Good for NEW runs; NOT for reproducing published arms, because a subset changes
# which documents exist and therefore which the shuffle index samples.
#
# Shared with everyone who can reach the fileset: both /proj/dmfexp and /proj/datasets are
# drwxrws--- root:proj_*, so group-readable is the maximum useful, and proj_dmfexp has 100+ members
# (bharat, psattig, ... ). `grp_ebm` is an LSF user group, not a POSIX group, so it cannot be used
# for file permissions.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
SRC=/proj/datasets/granite-4-datasets-megatron-merged
SHARED=/proj/dmfexp/datasets-shared/granite-4-cmix-subset
STAGED=/proj/dmfexp/nima/datasets-rescue          # the 142 GB already copied earlier
WEB_TOKENS=${WEB_TOKENS:-20e9}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null || true
mkdir -p "$SHARED"
rc=0
echo "[$(date -u +%FT%TZ)] start; dmfexp avail $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"

# 1. the two web shards, as prefix subsets
for b in web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1; do
  if [ -f "$SHARED/$b.bin" ] && [ -f "$SHARED/$b.idx" ]; then echo "  already built: $b"; continue; fi
  echo "  building subset: $b (${WEB_TOKENS} tokens)"
  python3 "$EXP/scripts/build_dataset_subset.py" --src "$SRC/$b" --dst "$SHARED/$b" \
      --tokens "$WEB_TOKENS" 2>&1 | grep -vE "INFO|WARNING|^W[0-9]" || rc=1
done

# 2. the math sets and indices -- move the copies already staged, else copy from source
for f in megamath-web-pro_0 finemath-3plus-rewritten_0; do
  for ext in bin idx ndocs; do
    [ -f "$SHARED/$f.$ext" ] && continue
    if [ -f "$STAGED/$f.$ext" ]; then
      mv "$STAGED/$f.$ext" "$SHARED/$f.$ext" && echo "  moved staged $f.$ext"
    elif [ -f "$SRC/$f.$ext" ]; then
      cp "$SRC/$f.$ext" "$SHARED/$f.$ext.part" && mv "$SHARED/$f.$ext.part" "$SHARED/$f.$ext" && echo "  copied $f.$ext"
    fi
  done
done

# 3. the tokenizer
mkdir -p "$SHARED/tokenizers"
[ -d "$SHARED/tokenizers/granite-4.0-tiktoken" ] || cp -a /proj/datasets/tokenizers/granite-4.0-tiktoken "$SHARED/tokenizers/" && echo "  tokenizer in place"

# 4. permissions: group-readable, setgid so anything added later inherits the group
chgrp -R proj_dmfexp "$SHARED" 2>/dev/null
find "$SHARED" -type d -exec chmod 2775 {} + 2>/dev/null
find "$SHARED" -type f -exec chmod 0664 {} + 2>/dev/null
chmod 2775 /proj/dmfexp/datasets-shared 2>/dev/null

echo "[$(date -u +%FT%TZ)] done rc=$rc; total $(du -sh "$SHARED" 2>/dev/null | cut -f1); dmfexp avail $(df -BG /proj/dmfexp | awk 'NR==2{print $4}')"
ls -l "$SHARED" | sed 's/^/    /'
exit $rc
