#!/bin/bash
# Report the IB-vs-TCP probe arms. Two traps this encodes, both of which bit me:
#  1. NEVER grep bare 'ncclRemoteError'/'IBV_WC_RETRY_EXC_ERR'. submit_train.sh's own comments and
#     its warning echo contain those strings, and the generated job script is echoed into the log,
#     so a bare grep reports 4 "errors" on a job that completed cleanly. Require the NCCL line
#     format: a real NCCL message carries "host:pid:tid" or "NCCL WARN".
#  2. Split DENSE (step < sparse_start_step) from SPARSE (>=). sparse_forward changes the
#     compute/comms ratio, so a median across the boundary compares two different things. And drop
#     any arm in SSUSP: suspension distorts the step times around it.
for j in "$@"; do
  id=${j%%:*}; nm=${j##*:}
  e=$HOME/bsub_logs/${nm}_${id}.stderr; o=$HOME/bsub_logs/${nm}_${id}.stdout
  [ -f "$e" ] || { printf "  %-14s (no log)\n" "$nm"; continue; }
  err=$(grep -hE 'IBV_WC_RETRY_EXC_ERR|ncclRemoteError' "$e" "$o" 2>/dev/null \
        | grep -E '[a-z0-9-]+:[0-9]+:[0-9]+|NCCL WARN' | grep -vc '^#')
  tr=$(grep -hm1 -oE 'NET/IB|NET/Socket' "$e" "$o" 2>/dev/null | head -1)
  st=$(bjobs -o stat -noheader "$id" 2>/dev/null || echo DONE)
  h=$(bjobs -o exec_host -noheader "$id" 2>/dev/null | tr -d ' ')
  # pair step number with step time off the same line, then split at 300
  # match the step lines by the two fields they uniquely share; do NOT use \xNN escapes (grep -E
  # does not interpret them) and do NOT use awk asort (not in mawk). Median via sort + awk.
  pairs=$(grep -oE 'step = [0-9]+,.*step_time \(sec\) = [0-9.]+' "$e" 2>/dev/null \
          | sed -E 's/step = ([0-9]+),.*step_time \(sec\) = ([0-9.]+).*/\1 \2/')
  med() { sort -n | awk '{a[NR]=$1} END{if(NR)printf "%.4f (n=%d)", a[int(NR/2)+1], NR; else printf "-"}'; }
  dmed=$(awk '$1>20 && $1<300 {print $2}' <<<"$pairs" | med)
  smed=$(awk '$1>=300 {print $2}' <<<"$pairs" | med)
  printf "  %-14s %-22s %-11s %-6s dense=%-18s sparse=%-18s err=%s\n" \
     "$nm" "${h:-?}" "${tr:-?}" "$st" "${dmed:--}" "${smed:--}" "$err"
done
