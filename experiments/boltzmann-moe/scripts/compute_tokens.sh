#!/bin/bash
# ============================================================================
# AUTHORITATIVE TOKEN CALCULATION for distributed training.
# This is THE formula. Use it. Do not hand-calculate.
#
#   tokens = seq_length × micro_batch_size × gradient_accumulation × total_GPUs × num_train_steps
#
# GPUS IS NOT IN THE CONFIG. It comes from the launcher (submit_train.sh).
# The config only has mbs, ga, seq, and steps. You MUST know the GPU count.
#
# Usage:
#   bash compute_tokens.sh <seq> <mbs> <ga> <gpus> <steps>
#   bash compute_tokens.sh --config <yml> <gpus>
# ============================================================================
set -euo pipefail

if [ "${1:-}" = "--config" ]; then
    cfg="$2"; gpus="${3:-8}"
    seq=$(grep "sequence_length:" "$cfg" | awk '{print $2}')
    mbs=$(grep "micro_batch_size:" "$cfg" | awk '{print $2}')
    ga=$(grep "gradient_accumulation_steps:" "$cfg" | awk '{print $2}')
    steps=$(grep "num_training_steps:" "$cfg" | awk '{print $2}')
    echo "Config: $(basename $cfg)"
else
    seq="$1"; mbs="$2"; ga="$3"; gpus="$4"; steps="$5"
fi

tok_per_step=$((seq * mbs * ga * gpus))
total=$((tok_per_step * steps))

echo "  seq=$seq  mbs=$mbs  ga=$ga  GPUs=$gpus  steps=$steps"
echo "  tok/step = $seq × $mbs × $ga × $gpus = $tok_per_step"
echo "  total = $tok_per_step × $steps = $total ($(python3 -c "print(f'{$total/1e9:.2f}B')"))"
