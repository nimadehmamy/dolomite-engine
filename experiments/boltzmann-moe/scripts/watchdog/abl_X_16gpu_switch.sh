#!/bin/bash
# Wait for abl_AA to finish, then retire abl_X's 8-GPU job so the watchdog restarts it at 16.
#
# WHY AN LSF JOB and not a cron or a nohup: it must outlive the interactive session. CLAUDE.md
# records that the token-milestone cadence was once a session-scoped CronCreate job that "died
# silently with its session" and cost the 8B and 16B anchors of five arms. The watchdog survives
# because it self-resubmits as an LSF job; this does the same.
#
# WHAT IT DOES, and nothing else:
#   wait until abl_AA's latest_checkpointed_iteration >= 122070   (genuinely finished, not crashed)
#   then bkill the abl_X job named abl_X_400M_rnorm
#   the watchdog's next poll sees abl_X not alive and below its target step, and resubmits it from
#   watchdog_jobs.conf -- which by then points at the _16gpu config with gpus=16 (8/node x 2 nodes).
# It requests NO GPUs. It is idempotent via a sentinel file.
#
# SAFETY: it waits on the CHECKPOINT reaching target, not on the job leaving the queue. If abl_AA
# crashes and the watchdog resubmits it, the checkpoint will not be at target and this keeps
# waiting -- which is the correct behaviour. It also refuses to act if abl_X is already within
# 2000 steps of done (no point) or is not running (nothing to kill).
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
AA_JSON=$EXP/results/iclr26_abl/abl_AA_134M_sandwich_isoall/latest_checkpointed_iteration.json
X_SP=$EXP/results/cmix/abl_X_400M_hyb_rnorm_none
AA_TARGET=122070
X_TARGET=61035
SENTINEL=$X_SP/.switched_to_16gpu
LOG=$EXP/scripts/watchdog/abl_X_16gpu_switch.log
START_TS=$(date +%s)
WALL_HHMM=${WALL_HHMM:-12:00}
WALL_SECONDS=$(( ${WALL_HHMM%%:*} * 3600 + ${WALL_HHMM##*:} * 60 ))
CONF=$EXP/scripts/watchdog/watchdog_jobs.conf
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

[ -f "$SENTINEL" ] && { log "sentinel present; already switched. exiting."; exit 0; }
log "started. waiting for abl_AA ckpt >= $AA_TARGET"

step_of(){ python3 -c "import json;print(json.load(open('$1'))['latest_checkpointed_iteration'])" 2>/dev/null || echo 0; }

while true; do
    aa=$(step_of "$AA_JSON")
    if [ "${aa:-0}" -ge "$AA_TARGET" ] 2>/dev/null; then
        log "abl_AA reached $aa >= $AA_TARGET"
        # the conf MUST already point at 16 GPUs, else killing abl_X just restarts it at 8
        if ! grep -qE '^abl_X_400M_rnorm\|.*_16gpu\.yml\|.*\|16\|' "$CONF"; then
            log "REFUSING: watchdog conf does not yet name the _16gpu config with gpus=16. Not killing."
            exit 1
        fi
        xstep=$(step_of "$X_SP/latest_checkpointed_iteration.json")
        if [ "${xstep:-0}" -ge $((X_TARGET-2000)) ] 2>/dev/null; then
            log "abl_X already at $xstep of $X_TARGET; not worth switching. exiting."
            touch "$SENTINEL"; exit 0
        fi
        jid=$(bjobs -noheader -o "jobid" -J abl_X_400M_rnorm 2>/dev/null | head -1 | tr -d ' ')
        if [ -z "$jid" ]; then
            log "no live abl_X job to kill; the watchdog will pick it up at 16 GPUs on its own."
            touch "$SENTINEL"; exit 0
        fi
        log "killing abl_X job $jid (at step $xstep) so the watchdog resubmits it at 16 GPUs"
        bkill "$jid" 2>&1 | tee -a "$LOG"
        sleep 45
        st=$(bjobs -noheader -o "stat" "$jid" 2>/dev/null | tr -d ' ')
        if [ -n "$st" ] && [ "$st" = "RUN" ]; then
            log "still RUN after plain bkill -- escalating to bkill -r (documented: a wedged job ignores SIGTERM)"
            bkill -r "$jid" 2>&1 | tee -a "$LOG"; sleep 30
        fi
        log "final state of $jid: $(bjobs -noheader -o 'stat' "$jid" 2>/dev/null | tr -d ' ' || echo GONE)"
        touch "$SENTINEL"
        log "done. watchdog will resubmit abl_X at 16 GPUs within one poll cycle."
        exit 0
    fi
    log "abl_AA at ${aa:-?}/$AA_TARGET -- waiting"
    # SELF-RESUBMIT before our own walltime ends, and also if we are preempted we simply are not
    # here -- so the watchdog is not the only thing that must outlive the session; this must too.
    # Mirrors watchdog_loop.sh's own pattern (SELF_RESUBMIT_BUFFER).
    if [ $(( $(date +%s) - START_TS )) -ge $(( WALL_SECONDS - 900 )) ]; then
        log "approaching walltime; resubmitting self"
        bsub -q preemptable -G grp_preemptable -J abl_X_16gpu_switch -n 1 -M 4G -W "$WALL_HHMM" \
             -o "$HOME/bsub_logs/abl_X_16gpu_switch_%J.stdout" \
             -e "$HOME/bsub_logs/abl_X_16gpu_switch_%J.stderr" \
             "bash $EXP/scripts/watchdog/abl_X_16gpu_switch.sh" 2>&1 | tee -a "$LOG"
        exit 0
    fi
    sleep 300
done
