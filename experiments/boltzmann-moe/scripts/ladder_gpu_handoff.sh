#!/bin/bash
# ⛔ DEAD 2026-09-22 -- DO NOT RESUBMIT. A 1-GPU checkpoint cannot load at 8 GPUs
#    (lr_scheduler/rng_state are plain per-rank .pt files; only model/optimizer are DCP).
#    See HANDOFF 17.11. Kept for the record only.
# ladder_gpu_handoff.sh -- promote the prefactor-ladder arms from 1 GPU to 8 GPUs as the
# grp_ebm arms finish and release their 8-GPU blocks.
#
# WHY. The ladder (abl_J/K/L) is running at 1 GPU because that was the only shape obtainable
# on 2026-09-21 (8-GPU, 2-GPU and 8x1-spread all PENDed; only standalone 1-GPU jobs ran).
# At 1 GPU the arms need ~18.6 h to reach 4B tokens; at 8 GPUs, ~2.3 h. The four grp_ebm arms
# release 8 GPUs each as they finish (H_6G6E, H_6G6S, h400M, sandwich), so each release can
# promote one ladder arm.
#
# ISO-TOKEN IS PRESERVED BY CONSTRUCTION: the 1-GPU configs are mbs 2 / ga 32 and the 8-GPU
# configs are mbs 2 / ga 4, both exactly 1*2*32*4096 = 8*2*4*4096 = 262,144 tok/step. So the
# promotion does not perturb the LR schedule or the token budget, and the resumed run is
# directly comparable to the finished sqrt_consistent reference step-for-step.
#
# ORDERING, AND WHY IT IS THIS WAY ROUND. Two jobs writing one save_path can race on
# latest_checkpointed_iteration.json and on max_to_keep pruning -- pre-flight rule 5. But
# killing the 1-GPU job FIRST and then submitting risks the freed GPUs being taken by someone
# else, leaving the arm with NOTHING running. So: submit the 8-GPU job, poll tightly until it
# is RUN, and only then kill the 1-GPU twin. The overlap is one 15 s poll, and it falls inside
# the 8-GPU job's compile phase (>100 s) during which it writes no checkpoint.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
SELF=$EXP/scripts/ladder_gpu_handoff.sh
LOG=$EXP/scripts/ladder_gpu_handoff.log
STATE=$EXP/scripts/ladder_gpu_handoff.state
touch "$STATE"
CYCLE=${CYCLE:-120}
START=$(date +%s); WALL=${AUTO_WALL:-84000}; BUF=1800
log(){ echo "[$(date -u '+%F %T')Z] $*" >> "$LOG"; }

# Priority order REVISED AGAIN 2026-09-21 23:13Z to K, L, J (was L, K, J, was L, J, K).
# RULE: give the scarce NON-PREEMPTABLE GPUs to the arm that cannot make progress without them,
# not to the one that is already advancing.
#   K (inv_sqrt, 16.7x) FIRST -- preempted THREE times in 20 min and net-NEGATIVE: 2950 -> rolled
#                                back to 2030 -> 2090 -> suspended again. It cannot finish on
#                                preemptable. It is also the magnitude midpoint of the ladder.
#   L (exact, 1.0x)     second -- the decisive arm, but it is advancing fine on preemptable
#                                (3590 and climbing), so it does not need rescuing yet.
#   J (mean, 1.0x)      last   -- least informative: J and L are both 1x and differ only in
#                                gelu' fidelity (cos 0.9968 vs 1.0), measured as a 1.46x vs 1.49x
#                                step-shift against K, i.e. no resolvable difference.
ARMS="abl_K_gsInvSqrt:abl_K_gsInvSqrt_1gpu:abl_K_134M_pure_1x12E_gsInvSqrt
abl_L_gsExact:abl_L_gsExact_1gpu:abl_L_134M_pure_1x12E_gsExact
abl_J_gsMean:abl_J_gsMean_1gpu:abl_J_134M_pure_1x12E_gsMean"

log "=== handoff watcher start (pid=$$ host=$(hostname) jid=${LSB_JOBID:-none}) ==="
while true; do
    el=$(( $(date +%s) - START ))
    if [ $el -ge $((WALL - BUF)) ]; then
        log "walltime near (${el}s); self-resubmitting"
        bsub -q preemptable -G grp_preemptable -J boltz_ladder_handoff -n 1 -M 4G -W 24:00 \
             -o "$HOME/bsub_logs/boltz_ladder_handoff_%J.stdout" \
             -e "$HOME/bsub_logs/boltz_ladder_handoff_%J.stderr" "bash $SELF" >> "$LOG" 2>&1
        log "exiting for successor"; exit 0
    fi

    used=$(blimits 2>/dev/null | awk '/grp_ebm/ {print $NF}' | tr -d ' ' | cut -d/ -f1)
    used=${used:-32}
    free=$(( 32 - used ))

    if [ "$free" -ge 8 ]; then
        while IFS=: read -r name8 name1 cfg8; do
            [ -z "${name8:-}" ] && continue
            grep -qx "$name8" "$STATE" && continue                       # already promoted
            j1=$(bjobs -noheader -o "jobid job_name stat" 2>/dev/null | awk -v n="$name1" '$2==n && $3=="RUN" {print $1}' | head -1)
            [ -z "$j1" ] && continue                                     # no 1-GPU twin running
            log "grp_ebm has $free free GPUs; promoting $name8 (1-GPU job $j1)"
            out=$(bash "$EXP/scripts/bsub/submit_train.sh" "$name8" \
                  "$REPO/configs/iclr_26/ablations/${cfg8}.yml" 8 ebm 24:00 200G 2>&1)
            j8=$(echo "$out" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+" | head -1)
            if [ -z "$j8" ]; then log "  submit FAILED, leaving 1-GPU job alone: $(echo "$out"|tail -2)"; break; fi
            log "  submitted 8-GPU job $j8; waiting for it to reach RUN before killing $j1"
            ok=0
            for _ in $(seq 1 80); do          # up to 20 min at 15 s
                st=$(bjobs -o stat -noheader "$j8" 2>/dev/null)
                if [ "$st" = "RUN" ]; then ok=1; break; fi
                if [ "$st" = "EXIT" ] || [ -z "$st" ]; then log "  8-GPU job $j8 went $st; ABORT, 1-GPU job $j1 keeps running"; break; fi
                sleep 15
            done
            if [ "$ok" = 1 ]; then
                # DEREGISTER THE 1-GPU ENTRY FIRST. As of 2026-09-21 the *_1gpu arms are in
                # watchdog_jobs.conf so a preemption auto-resumes them. But the watchdog reads
                # "job gone + step < target" as a death, so the kill below would make it
                # resubmit a 1-GPU twin that races the 8-GPU job on the SAME save_path --
                # pre-flight rule 5, two writers pruning one checkpoint set. Comment the line
                # out BEFORE killing, and drop any state row so it is not re-evaluated.
                WDC="$EXP/scripts/watchdog/watchdog_jobs.conf"
                WDS="$EXP/scripts/watchdog/watchdog_state.txt"
                if grep -q "^${name1}|" "$WDC" 2>/dev/null; then
                    # Prefix the WHOLE line with '#'. An earlier version replaced just the
                    # name, which left a line starting with '|' -- an entry with an EMPTY name,
                    # skipped only by the loop's `[[ -z "${name// }" ]]` guard. Too fragile.
                    sed -i "/^${name1}|/s|^|# PROMOTED-TO-8GPU $(date -u +%FT%TZ) |" "$WDC"
                    log "  deregistered $name1 from watchdog_jobs.conf"
                fi
                sed -i "/^${name1} /d" "$WDS" 2>/dev/null
                log "  8-GPU job $j8 is RUN; killing 1-GPU twin $j1 now"
                bkill "$j1" >> "$LOG" 2>&1
                echo "$name8" >> "$STATE"
                log "  promoted $name8 -> 8 GPUs (job $j8); 1-GPU $j1 killed"
            else
                log "  giving up on this cycle; killing the pending 8-GPU job $j8 to free the slot"
                bkill "$j8" >> "$LOG" 2>&1
            fi
            break                              # at most one promotion per cycle
        done <<< "$ARMS"
    fi
    sleep "$CYCLE"
done
