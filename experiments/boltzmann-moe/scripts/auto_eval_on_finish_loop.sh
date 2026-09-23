#!/bin/bash
# auto_eval_on_finish_loop.sh -- CPU-only babysitter for auto_eval_on_finish.sh.
#
# WHY THIS EXISTS (2026-09-21). The eval of a finished cmix/iclr_26 arm was being triggered BY
# HAND from a Claude session. Sessions die: on 2026-09-21 a session exit silently killed four
# Monitor watches, and the h400M completion trigger died with them. Separately, the long-running
# babysitter `boltz_auto_eval` does NOT cover these arms -- it drives
# collect_flops_wave_20260912.sh, whose CFGDIRS list is the eight configs/iclr_* dirs and
# includes NEITHER configs/cmix NOR configs/iclr_26. So no unattended process would ever have
# evaluated cmix_400M_hybrid_sparse, cmix_400M_sandwich_sparse or the abl_H pair.
#
# auto_eval_on_finish.sh is EDGE-TRIGGERED by its own design (it fires only when
# latest_checkpointed_iteration == num_training_steps, the results file is absent, AND no live
# bsub job of that name exists), so calling it on a short cycle is safe and cannot double-submit.
#
# QUEUE: the evals it submits go to preemptable, submit_gpu_test.sh's default. Do NOT set
# EVAL_QUEUE=ebm here: grp_ebm is a 32-GPU allocation that routinely sits at 32/32, and an eval
# submitted there PENDS INDEFINITELY (two were found stuck 2-4 h on 2026-09-16). Preemptable
# always schedules, and auto_eval_on_finish.sh already splits gsm8k_cot into its own job so a
# preemption costs one task rather than a >1.5 h eval.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
SELF=$EXP/scripts/auto_eval_on_finish_loop.sh
LOG=$EXP/scripts/auto_eval_on_finish_loop.log
CYCLE=${CYCLE:-300}
START=$(date +%s); WALL=${AUTO_WALL:-84000}; BUF=1800
log(){ echo "[$(date -u '+%F %T')Z] $*" >> "$LOG"; }
log "=== start (pid=$$ host=$(hostname) jid=${LSB_JOBID:-none}) ==="
while true; do
    el=$(( $(date +%s) - START ))
    if [ $el -ge $((WALL - BUF)) ]; then
        log "walltime near (${el}s); self-resubmitting"
        bsub -q preemptable -G grp_preemptable -J boltz_eval_finish -n 1 -M 4G -W 24:00 \
             -o "$HOME/bsub_logs/boltz_eval_finish_%J.stdout" \
             -e "$HOME/bsub_logs/boltz_eval_finish_%J.stderr" "bash $SELF" >> "$LOG" 2>&1
        log "exiting for successor"; exit 0
    fi
    out=$(bash "$EXP/scripts/auto_eval_on_finish.sh" 2>&1 | grep -E "submitted|skip" || true)
    [ -n "$out" ] && log "$out"
    sleep "$CYCLE"
done
