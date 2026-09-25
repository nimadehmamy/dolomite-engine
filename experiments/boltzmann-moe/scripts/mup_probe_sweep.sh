#!/bin/bash
# µP probe sweep: 4 widths × 5 LRs = 20 short runs
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CFG_DIR=$REPO/configs/iclr_26/fwe_sweep/mup_probes
RESULTS=$REPO/experiments/boltzmann-moe/results/fwe/mup_probes
DATA_PREFIX="/proj/dmfexp/datasets-shared/granite-4-cmix-FULL/web-nemotron-cc-hq-p2_0"
CACHE="/proj/dmfexp/nima/.cache/megatron_fwe_nemotron"
TOKENIZER="/proj/datasets/tokenizers/granite-4.0-tiktoken"

mkdir -p "$CFG_DIR" "$RESULTS"

STEPS=500
WARMUP=50
DECAY=$((STEPS - WARMUP))

# Widths and their derived params
# d, heads, I_mlp, I_e, K
WIDTHS=(
    "256 4 704 1152 8"
    "384 6 1088 1728 8"
    "512 8 1408 2304 8"
    "768 12 2112 3392 8"
)

LRS="1e-2 5e-3 2e-3 1e-3 5e-4"

GPUS_PER=2  # 2 GPUs per probe (tiny models)
QUEUE="${1:-ebm}"  # default grp_ebm

for width_spec in "${WIDTHS[@]}"; do
    read -r d heads I_mlp I_e K <<< "$width_spec"
    I_total=$((K * I_e))
    
    for lr in $LRS; do
        name="mup_d${d}_lr${lr}"
        cfg="$CFG_DIR/${name}.yml"
        
        cat > "$cfg" << YMLEOF
# µP probe: d=$d, lr=$lr, $STEPS steps
datasets:
- class_name: MegatronDataset
  data_name: Megatron
  data_sampling_ratio: 1
  class_args:
    eval_steps: 2
    data_cache_path: $CACHE
    data_path:
    - 1.0
    - $DATA_PREFIX
    split: 99,0.5,0.5
    sequence_length: 4096
tokenizer_args:
  tokenizer_name: $TOKENIZER
model_args:
  model_class: AutoModelForCausalLM
  pretrained_config:
    model_type: energy
    num_iterations: 1
    num_pre_layers: 0
    num_post_layers: 0
    layer_iterations: [1,1,1,1,1,1,6]
    initializer_range: 0.02
    layer_norm_epsilon: 1.0e-05
    normalization_function: rmsnorm
    position_embedding_type: rope
    rope_dim: 64
    hidden_size: $d
    num_layers: 7
    init_method: normal
    tie_word_embeddings: true
    energy_proj_type: psd_anti
    bos_token_id: 100257
    eos_token_id: 100257
    pad_token_id: 100256
    vocab_size: 100352
    max_position_embeddings: 4096
    sequence_mixer_blocks:
    - &gpt_attn {sequence_mixer_type: softmax_attention, num_attention_heads: $heads, num_key_value_heads: $heads, add_bias: false, attention_multiplier: 0.125}
    - *gpt_attn
    - *gpt_attn
    - *gpt_attn
    - *gpt_attn
    - *gpt_attn
    - sequence_mixer_type: energy_attention
      num_attention_heads: $heads
      num_key_value_heads: $heads
      add_bias: false
      attention_multiplier: 0.125
    mlp_blocks:
    - &gpt_ffn {mlp_type: MLP, activation_function: swiglu, intermediate_size: $I_mlp, add_bias: false}
    - *gpt_ffn
    - *gpt_ffn
    - *gpt_ffn
    - *gpt_ffn
    - *gpt_ffn
    - mlp_type: EnergyFF_BoltzmannMoE
      expert_kind: w1w2
      intermediate_size: $I_total
      n_experts: $K
      top_k: 2
      temperature: 1.0
      e_sign_override: neg
      repulsion_coef: 0.1
      repulsion_form: abs
      repulsion_space: output
      repulsion_subsample: 64
      repulsion_tensor_idx: true
      n_repulsion_pairs: 4
      gelu_grad_method: sigmoid
      fused_experts: true
      sparse_forward: true
      sparse_start_step: 100
      sparse_candidates: 2
      sparse_explore: 2
      proxy_loss_coef: 0.01
      proxy_kind: subspace
      proxy_rank: 16
      proxy_out_dim: 512
      sinkhorn_iters: 3
      sinkhorn_persist_mu: true
      sinkhorn_mu_iters: 6
      renormalize_topk: true
      routing_norm: none
      activation_function: gelu
      add_bias: false
tuning_args:
  tuning_method: pretraining
save_args:
  save_path: $RESULTS/$name
  save_interval: 10000
  max_to_keep: 1
logging_args:
  log_interval: 10
  experiments_tracker_name: wandb
  wandb_args:
    project: boltzmann-moe-fwe
    name: $name
training_parameters:
  num_training_steps: $STEPS
  eval_interval: 10000
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  eval_during_training: false
  gradient_clipping: 1
optimizer_args:
  class_name: TorchAdamW
  class_args:
    lr: $lr
    weight_decay: 0.1
    betas: [0.9, 0.95]
    eps: 1.0e-08
lr_scheduler_args:
  lr_decay_style: cosine
  lr_decay_factor: 0.1
  num_warmup_steps: $WARMUP
  num_constant_steps: 0
  num_decay_steps: $DECAY
mixed_precision_args:
  dtype: bf16
distributed_args:
  fsdp_algorithm: 2
  gradient_checkpointing_method: block
  torch_compile: true
  stage: 0
YMLEOF
        echo "  Generated $name"
    done
done

echo ""
echo "Generated $(ls $CFG_DIR/*.yml 2>/dev/null | wc -l) probe configs in $CFG_DIR"
echo ""
echo "To submit all probes (2 GPUs each):"
echo "  for cfg in $CFG_DIR/mup_d*.yml; do"
echo "    name=\$(basename \$cfg .yml)"
echo "    bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \$name \$cfg 2 ebm 01:00 32G"
echo "  done"
