# FineWeb-Edu Sweep Results (2026-09-24 to 2026-09-25)

## ⚠ DATA NOTE: despite the "fineweb" prefix, all arms trained on nemotron-cc

The configs are named `fineweb_*` but the actual training data is
`/proj/dmfexp/datasets-shared/granite-4-cmix-FULL/web-nemotron-cc-hq-p2_0`
(a single web shard from the granite-4 corpus, granite-4.0 tiktoken tokenizer).

FineWeb-Edu was the original plan but its `.idx` files used an incompatible
Megatron format variant (LLMB: version=1, dtype_code=0, doc_count field holds
total_tokens). The reader was patched on 2026-09-25 and a trial run started,
but no full arm has completed on actual FineWeb-Edu data.

All arms in this directory trained on the SAME data shard, so the comparisons
are internally valid. The data is 100% web (no math), unlike the cmix blend
(70% web / 30% math) used in the megatron-era experiments (HANDOFF §1-28).

## Directory structure

```
results/fwe/
├── fineweb_<arm>/                  # one directory per arm
│   ├── global_step<N>/            # DCP checkpoint (model + optimizer)
│   ├── latest_checkpointed_iteration.json
│   ├── unsharded_step<N>/         # unsharded HF checkpoint + eval results
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── harness_results_<timestamp>.json       # raw eval harness output
│   │   ├── harness_results_merged_<timestamp>.json # merged (likelihood + gsm8k)
│   │   └── <munged_path>/results_<timestamp>.json  # alternative output path
│   └── sparseeval_step<N>/        # eval with proxy router (energy arms only)
│       ├── harness_results_<timestamp>.json
│       └── <munged_path>/results_<timestamp>.json
└── mup_probes/                    # µP scaling probes (500 steps each)
    └── mup_d<width>_lr<rate>/
```

## JSON structure

Each `harness_results_*.json` or `results_*.json` contains:
```json
{
  "results": {
    "<task_name>": {
      "acc,none": 0.xxx,        // accuracy (for boolq, copa, winogrande, race, lambada)
      "acc_norm,none": 0.xxx,   // normalized accuracy (for arc, hellaswag, openbookqa, piqa, sciq)
      ...
    },
    "mmlu": {"acc,none": 0.xxx},
    "wikitext": {"word_perplexity,none": xx.x},
    ...
  },
  "config": { ... }
}
```

The `_merged` files are created by `merge_eval_results.py` and combine the
likelihood eval with a separately-run GSM8K-CoT eval. For this sweep, GSM8K
was skipped, so the merged file is identical to the raw one.

## Key scripts

| script | purpose |
|---|---|
| `experiments/eval_scripts/compute_avg11.py <run_dir>` | Compute Avg11 from a single arm's eval JSON |
| `experiments/boltzmann-moe/scripts/gen_table1_fwe.py` | Generate the FineWeb results table (text + LaTeX) |
| `lm_engine/hf_models/modeling_utils/mlp_blocks/energy_ff_paramcount.py` | `audit_config(path)` → total/active/FLOPwt from YAML |

## Avg11 recipe

11-task unweighted mean:
- `acc_norm`: arc_challenge, arc_easy, hellaswag, openbookqa, piqa, sciq
- `acc`: boolq, copa, winogrande, race, lambada_openai

MMLU (acc) and GSM8K-CoT (flex-extract) are reported SEPARATELY, never averaged in.

## Reproducing the table

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
python experiments/boltzmann-moe/scripts/gen_table1_fwe.py
```

## Parameter verification

```bash
python -c "
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
r = audit_config('configs/iclr_26/fwe_sweep/<arm>.yml')
print(r)
"
```
