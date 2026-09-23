#!/bin/bash
# compare_prefactor_ladder.sh -- matched-step comparison of the hopfield_grad_scale ladder.
#
# All four arms share one parent (configs/cmix/cmix_134M_pure_32B_sparse.yml), one seed (42,
# the default -- set in no config), and 262,144 tok/step, so matched-step lm_loss differences
# are attributable to the prefactor alone. lm_loss is in NATS, LOWER IS BETTER.
#
# DO NOT READ THIS BEFORE STEP 2000. Steps 1-2000 are LR warmup and the sparse path only
# activates at step 300; the step-300..400 ordering on 2026-09-21 showed the 16.7x arm ahead by
# 0.90 nats, which is exactly the regime where early descent speed does not predict final
# quality. The milestones worth quoting are step 4000 (1.049B), 8000 (2.097B) and 16000 (4.194B).
set -uo pipefail
declare -A JOB=( [L_exact_1.0x]=1829569 [J_mean_1.0x]=1830287 [K_invsqrt_16.7x]=1830288 )
ORDER="L_exact_1.0x J_mean_1.0x K_invsqrt_16.7x"
STEPS="${*:-500 1000 2000 3000 4000 8000 16000}"
for nm in $ORDER; do
  j=${JOB[$nm]}
  e=$(ls -t $HOME/bsub_logs/*_${j}.stderr 2>/dev/null | head -1)
  : > /tmp/ladder_$j.txt
  [ -n "$e" ] && grep -oE "step = [0-9]+, train-loss = [0-9.]+, train-lm_loss = [0-9.]+" "$e" \
      | sed -E 's/step = ([0-9]+).*lm_loss = ([0-9.]+)/\1 \2/' | sort -n -u -k1,1 > /tmp/ladder_$j.txt
done
echo "lm_loss (nats, LOWER better) -- iso-token 262,144 tok/step, same seed, same data order"
printf "  %-8s" step; for nm in $ORDER; do printf " %16s" "$nm"; done; echo
for s in $STEPS; do
  printf "  %-8s" "$s"
  for nm in $ORDER; do
    v=$(awk -v s="$s" '$1==s{print $2}' /tmp/ladder_${JOB[$nm]}.txt)
    printf " %16s" "${v:--}"
  done; echo
done
echo
echo "reference (sqrt_consistent, 66.9x) from its finished 32B log: 4.3486 @2000, 3.7345 @5000, 3.4915 @10000, 3.4180 @15260(4.00B), 3.3344 @30520(8.00B)"
echo
echo "milestone anchors preserved so far (benchmarkable, not just loss):"
for nm in $ORDER; do
  case $nm in L_*) d=abl_L_134M_pure_1x12E_gsExact;; J_*) d=abl_J_134M_pure_1x12E_gsMean;; K_*) d=abl_K_134M_pure_1x12E_gsInvSqrt;; esac
  p=/proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/results/iclr26_abl/$d/milestones
  printf "  %-16s %s\n" "$nm" "$(ls "$p" 2>/dev/null | tr '\n' ' ' || echo none)"
done
