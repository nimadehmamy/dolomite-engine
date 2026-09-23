#!/bin/bash
# Copy the FULL corpus to /proj/dmfexp so it survives the owners' deletion AND a fileset purge.
#
# WHY, beyond the hard links. /proj/datasets/ndehmamy-dataset-rescue/ holds the full .bin+.idx as HARD
# LINKS (links=2, same inode, dev=54). That defends against the owners' `rm` but NOT against:
#   * in-place truncation -- both names share one set of blocks;
#   * destruction of the `datasets` fileset, which both paths sit in.
# This is a real second copy on a DIFFERENT fileset (dmfexp), so it is independent of both.
#
# Source is the rescue dir because its .bin and .idx are correctly PAIRED. Never mix
# full-corpus-indices/*.idx with a subset .bin (reads past EOF) or a subset .idx with a full .bin
# (silently trains on the first 18.6%) -- neither errors.
set -uo pipefail
SRC=/proj/datasets/ndehmamy-dataset-rescue
DST=/proj/dmfexp/datasets-shared/granite-4-cmix-FULL
SETS="web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1 megamath-web-pro_0 finemath-3plus-rewritten_0"
mkdir -p "$DST"
echo "$(date -u +%FT%TZ) START  free on dmfexp: $(df -h /proj/dmfexp | awk 'NR==2{print $4}')"
fail=0
for f in $SETS; do
  for e in bin idx ndocs; do
    s="$SRC/$f.$e"; d="$DST/$f.$e"
    [ -f "$s" ] || { echo "  MISSING SOURCE $s"; fail=1; continue; }
    ss=$(stat -c %s "$s")
    if [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$ss" ]; then
      echo "  skip $f.$e (already $ss bytes)"; continue
    fi
    echo "$(date -u +%H:%M:%SZ)   copying $f.$e ($(echo "scale=2;$ss/1073741824"|bc) GiB)"
    # copy to a .part name then rename, so an interrupted copy can never look complete
    cp --sparse=never "$s" "$d.part" || { echo "  COPY FAILED $f.$e"; fail=1; rm -f "$d.part"; continue; }
    ds=$(stat -c %s "$d.part")
    if [ "$ds" != "$ss" ]; then echo "  SIZE MISMATCH $f.$e: $ds vs $ss"; fail=1; rm -f "$d.part"; continue; fi
    mv "$d.part" "$d"
    echo "$(date -u +%H:%M:%SZ)   done $f.$e"
  done
done
# tokenizers: small, and a run cannot start without them
[ -d "$SRC/tokenizers" ] && cp -rn "$SRC/tokenizers" "$DST/" 2>/dev/null
echo "$(date -u +%FT%TZ) SIZES verified below; fail=$fail"
for f in $SETS; do for e in bin idx; do
  a=$(stat -c %s "$SRC/$f.$e" 2>/dev/null); b=$(stat -c %s "$DST/$f.$e" 2>/dev/null)
  printf "  %-34s %-4s src=%-14s dst=%-14s %s\n" "$f" "$e" "${a:-NA}" "${b:-NA}" "$([ "$a" = "$b" ] && echo OK || echo MISMATCH)"
done; done
echo "$(date -u +%FT%TZ) free after: $(df -h /proj/dmfexp | awk 'NR==2{print $4}')"
exit $fail
