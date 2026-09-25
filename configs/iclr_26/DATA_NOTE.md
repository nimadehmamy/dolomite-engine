# DATA NOTE for configs/iclr_26/fwe_sweep/

## The "fineweb" configs train on nemotron-cc, NOT FineWeb-Edu

Despite the `fineweb_*` prefix, all configs in `fwe_sweep/` point at:
```
/proj/dmfexp/datasets-shared/granite-4-cmix-FULL/web-nemotron-cc-hq-p2_0
```
This is a single web shard from the granite-4 Nemotron-CC corpus (10.4B tokens,
granite-4.0 tiktoken tokenizer, V=100352).

### Why not FineWeb-Edu?
The FineWeb-Edu data at `/proj/datasets/archive/web_deduped/fineweb-edu/` uses
an LLMB variant of the Megatron `.idx` format (dtype_code=0, doc_count field
holds total_tokens, N+1 pointers instead of N). The reader was patched on
2026-09-25 (`dtype.py`, `indexed_dataset.py`) and a trial run started, but no
full arm has completed on actual FineWeb-Edu.

### The config names are historical, not the data source
The names were set when FineWeb-Edu was the plan. Renaming them would break
wandb run tracking and the launch ledger. All arms in this sweep use the SAME
data, so comparisons are internally valid.

### For future runs on actual FineWeb-Edu
1. Use `data_name: FineWebEdu` (not `Megatron`)
2. Point at `/proj/datasets/archive/web_deduped/fineweb-edu/<shard>`
3. Use a separate `data_cache_path: /proj/dmfexp/nima/.cache/megatron_fwe_real`
4. The reader patches in `dtype.py` and `indexed_dataset.py` handle the format
