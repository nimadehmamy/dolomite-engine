#!/bin/bash
# FAST parallel corpus copy. Replaces the single-stream throttled version.
#
# WHY THE FIRST ATTEMPT WAS SLOW (106 MiB/s): it ran `cp` under `ionice -c2 -n7` -- the LOWEST I/O
# priority -- as one stream with the default block size. GPFS wants large blocks and parallel
# streams. This version: one stream per file, bs=64M, NO ionice. File 1 RESUMES from its .part
# offset instead of recopying ~990 GiB.
set -uo pipefail
S=/proj/datasets/ndehmamy-dataset-rescue
D=/proj/dmfexp/datasets-shared/granite-4-cmix-FULL
one() {
  local f=$1 src="$S/$1.bin" dst="$D/$1.bin.part" fin="$D/$1.bin"
  local want; want=$(stat -c %s "$src")
  if [ -f "$fin" ] && [ "$(stat -c %s "$fin")" = "$want" ]; then echo "[$f] already complete"; return 0; fi
  local off=0; [ -f "$dst" ] && off=$(stat -c %s "$dst")
  # never trust a .part larger than the source
  [ "$off" -gt "$want" ] && { rm -f "$dst"; off=0; }
  echo "[$f] $(date -u +%H:%M:%SZ) resume at $off of $want ($(( (want-off)/1073741824 )) GiB to go)"
  if [ "$off" -gt 0 ]; then
    dd if="$src" of="$dst" bs=64M skip="$off" seek="$off" iflag=skip_bytes \
       oflag=seek_bytes conv=notrunc status=none || { echo "[$f] DD FAILED"; return 1; }
  else
    dd if="$src" of="$dst" bs=64M status=none || { echo "[$f] DD FAILED"; return 1; }
  fi
  local got; got=$(stat -c %s "$dst")
  if [ "$got" != "$want" ]; then echo "[$f] SIZE MISMATCH $got vs $want"; return 1; fi
  mv "$dst" "$fin"; echo "[$f] $(date -u +%H:%M:%SZ) DONE $want bytes"
}
# small files first so they land immediately, then the two 1 TB shards in parallel
for f in megamath-web-pro_0 finemath-3plus-rewritten_0; do
  for e in idx ndocs; do cp -n "$S/$f.$e" "$D/$f.$e" 2>/dev/null; done
done
for f in web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1; do
  for e in idx ndocs; do cp -n "$S/$f.$e" "$D/$f.$e" 2>/dev/null; done
done
cp -rn "$S/tokenizers" "$D/" 2>/dev/null
echo "$(date -u +%FT%TZ) small files + indices done; starting 4 parallel .bin streams"
pids=()
for f in web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1 megamath-web-pro_0 finemath-3plus-rewritten_0; do
  one "$f" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -u +%FT%TZ) ALL STREAMS ENDED rc=$rc"
for f in web-nemotron-cc-hq-p2_0 web-nemotron-cc-hq-p2_1 megamath-web-pro_0 finemath-3plus-rewritten_0; do
  for e in bin idx; do
    a=$(stat -c %s "$S/$f.$e" 2>/dev/null); b=$(stat -c %s "$D/$f.$e" 2>/dev/null)
    printf "  %-34s %-4s %s\n" "$f" "$e" "$([ "$a" = "$b" ] && echo "OK $a" || echo "MISMATCH src=$a dst=${b:-absent}")"
  done
done
exit $rc
