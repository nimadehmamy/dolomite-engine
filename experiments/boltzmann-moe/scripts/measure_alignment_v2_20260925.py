"""Measure output alignment across layers in the G1 6S6E model (v2).

Captures three signal types per layer:
  1. Attention output (sequence_mixer output)
  2. FF/MoE output (mlp block output)
  3. Full layer delta (hidden_states_out - hidden_states_in)

Fixes the v1 energy-attention hook issue by reading block._attn_out directly
instead of relying on a forward hook (the hook captured zeros for energy layers).

Writes results to a JSON file for the figure script.

Usage:
    bash experiments/boltzmann-moe/scripts/bsub/submit_gpu_test.sh measure_align_v2 \
        "python experiments/boltzmann-moe/scripts/measure_alignment_v2_20260925.py" 1 01:00 32G
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Register custom model types
import lm_engine.hf_models  # noqa: F401

CKPT_BASE = REPO / "experiments/boltzmann-moe/results/fwe/fineweb_G1_6S6E_deep_sparse_d768"
N_LAYERS = 12
ENERGY_LAYERS = list(range(6, 12))
STANDARD_LAYERS = list(range(0, 6))

N_SEQS = 100
SEQ_LEN = 512
VOCAB_SIZE = 100352
DEVICE = "cuda"

OUT_JSON = REPO / "experiments/boltzmann-moe/scripts/alignment_v2_results.json"


# ---------------------------------------------------------------------------
# Model loading (same as v1)
# ---------------------------------------------------------------------------

def load_model(ckpt_base: Path, device: str = "cuda") -> torch.nn.Module:
    import yaml

    iter_file = ckpt_base / "latest_checkpointed_iteration.json"
    step = json.load(open(iter_file))["latest_checkpointed_iteration"]
    print(f"Loading checkpoint at step {step}")

    ckpt_dir = ckpt_base / f"global_step{step}"
    model_dir = ckpt_dir / "model"
    config_path = ckpt_dir / "training_config.yml"

    with open(config_path) as f:
        train_cfg = yaml.safe_load(f)

    from transformers import AutoConfig
    config = AutoConfig.for_model(**train_cfg["model_args"]["pretrained_config"])

    from lm_engine.hf_models.models.energy.main import EnergyForCausalLM

    print("  Instantiating model on meta device ...")
    with torch.device("meta"):
        model = EnergyForCausalLM(config)
    model = model.to_empty(device="cpu")

    print(f"  Loading distcp from {model_dir} ...")
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    state: dict = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(str(model_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    flat = state["state"]

    PREFIX = "model."
    if all(k.startswith(PREFIX) for k in flat):
        print(f"  Stripping '{PREFIX}' prefix from {len(flat)} keys")
        flat = {k[len(PREFIX):]: v for k, v in flat.items()}

    missing, unexpected = model.load_state_dict(flat, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"  Model loaded on {device}, dtype=bf16")
    return model


# ---------------------------------------------------------------------------
# Hook-based capture for standard layers + block-level delta capture
# ---------------------------------------------------------------------------

class OutputCaptureV2:
    """Capture attention, MLP, and full-layer-delta outputs for all layers.

    For standard layers (0-5): hooks on sequence_mixer and mlp_block.
    For energy layers (6-11): hooks on ffwd; reads block._attn_out after forward.
    For all layers: hooks on the block itself to capture input and output hidden states.
    """

    def __init__(self, model: torch.nn.Module):
        self.attn_outputs: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self.block_inputs: dict[int, torch.Tensor] = {}
        self.block_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._model = model
        self._register(model)

    def _register(self, model: torch.nn.Module):
        for i in range(N_LAYERS):
            block = model.transformer.h[i]

            # Block-level hook: capture input (hidden_states) and output for delta
            self._hooks.append(
                block.register_forward_hook(self._make_block_hook(i))
            )

            if i in ENERGY_LAYERS:
                # Energy block: hook on ffwd for MLP output
                self._hooks.append(
                    block.ffwd.register_forward_hook(self._make_hook(self.mlp_outputs, i))
                )
                # Also try hooking block.attn to diagnose the zero issue
                self._hooks.append(
                    block.attn.register_forward_hook(self._make_hook(self.attn_outputs, i, tag=f"hook_L{i}"))
                )
            else:
                # Standard GPT block: hooks on sequence_mixer and mlp_block
                self._hooks.append(
                    block.sequence_mixer.register_forward_hook(self._make_hook(self.attn_outputs, i))
                )
                self._hooks.append(
                    block.mlp_block.register_forward_hook(self._make_hook(self.mlp_outputs, i))
                )

    @staticmethod
    def _make_hook(store: dict, layer_idx: int, tag: str = ""):
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                output = output[0]
            store[layer_idx] = output.detach()
        return hook

    def _make_block_hook(self, layer_idx: int):
        """Hook for the entire block: captures input and output hidden states."""
        def hook(_module, input_args, output):
            # input_args[0] is hidden_states
            inp = input_args[0] if isinstance(input_args, tuple) else input_args
            if isinstance(inp, torch.Tensor):
                self.block_inputs[layer_idx] = inp.detach()
            if isinstance(output, tuple):
                output = output[0]
            self.block_outputs[layer_idx] = output.detach()
        return hook

    def read_energy_attn_from_block(self):
        """Read _attn_out from energy blocks after forward pass.

        This bypasses the hook mechanism that returned zeros in v1.
        If the hook-captured value differs from _attn_out, we overwrite with _attn_out.
        """
        for i in ENERGY_LAYERS:
            block = self._model.transformer.h[i]
            attn_out = getattr(block, '_attn_out', None)
            if attn_out is not None:
                hook_val = self.attn_outputs.get(i)
                if hook_val is not None:
                    hook_norm = hook_val.float().norm().item()
                    direct_norm = attn_out.float().norm().item()
                    if abs(hook_norm - direct_norm) > 1e-3:
                        # They differ! Use the direct value.
                        pass
                self.attn_outputs[i] = attn_out.detach()

    def clear(self):
        self.attn_outputs.clear()
        self.mlp_outputs.clear()
        self.block_inputs.clear()
        self.block_outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Cosine similarity: consecutive-layer only
# ---------------------------------------------------------------------------

def consecutive_cos_sim(vecs: dict[int, torch.Tensor], layers: list[int]) -> list[float | None]:
    """Compute mean per-token cosine similarity for consecutive layer pairs."""
    result = []
    for a_idx in range(len(layers) - 1):
        la, lb = layers[a_idx], layers[a_idx + 1]
        if la not in vecs or lb not in vecs:
            result.append(None)
            continue
        va = vecs[la].float().reshape(-1, vecs[la].shape[-1])
        vb = vecs[lb].float().reshape(-1, vecs[lb].shape[-1])
        va_n = F.normalize(va, dim=-1)
        vb_n = F.normalize(vb, dim=-1)
        cos = (va_n * vb_n).sum(-1).mean().item()
        result.append(cos)
    return result


def compute_deltas(inputs: dict[int, torch.Tensor],
                   outputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Compute layer deltas: delta_i = output_i - input_i."""
    deltas = {}
    for i in range(N_LAYERS):
        if i in inputs and i in outputs:
            deltas[i] = outputs[i] - inputs[i]
    return deltas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  G1 6S6E alignment measurement (v2)")
    print("=" * 60)

    model = load_model(CKPT_BASE, device=DEVICE)
    capture = OutputCaptureV2(model)

    all_layers = list(range(N_LAYERS))
    batch_size = 4
    n_full_batches = N_SEQS // batch_size

    # Accumulators
    attn_cos_sums = [0.0] * (N_LAYERS - 1)
    mlp_cos_sums = [0.0] * (N_LAYERS - 1)
    delta_cos_sums = [0.0] * (N_LAYERS - 1)
    attn_counts = [0] * (N_LAYERS - 1)
    mlp_counts = [0] * (N_LAYERS - 1)
    delta_counts = [0] * (N_LAYERS - 1)

    print(f"\nRunning {N_SEQS} sequences of length {SEQ_LEN} ...")

    with torch.no_grad():
        for b_idx in range(n_full_batches):
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
            capture.clear()

            model(input_ids=input_ids)

            # Read energy attention outputs from block._attn_out
            capture.read_energy_attn_from_block()

            # Diagnostics on first batch
            if b_idx == 0:
                print("\n--- Diagnostics (batch 0) ---")
                for i in range(N_LAYERS):
                    attn_v = capture.attn_outputs.get(i)
                    mlp_v = capture.mlp_outputs.get(i)
                    blk_in = capture.block_inputs.get(i)
                    blk_out = capture.block_outputs.get(i)
                    tag = "E" if i in ENERGY_LAYERS else "S"
                    attn_info = f"norm={attn_v.float().norm().item():.2f} shape={tuple(attn_v.shape)}" if attn_v is not None else "MISSING"
                    mlp_info = f"norm={mlp_v.float().norm().item():.2f}" if mlp_v is not None else "MISSING"
                    in_info = f"norm={blk_in.float().norm().item():.2f}" if blk_in is not None else "MISSING"
                    out_info = f"norm={blk_out.float().norm().item():.2f}" if blk_out is not None else "MISSING"
                    delta_norm = f"{(blk_out - blk_in).float().norm().item():.2f}" if (blk_in is not None and blk_out is not None) else "N/A"
                    print(f"  L{i:2d} [{tag}]: attn={attn_info:>40s}  mlp={mlp_info:>12s}  "
                          f"delta_norm={delta_norm:>8s}")

            # Compute deltas
            deltas = compute_deltas(capture.block_inputs, capture.block_outputs)

            # Consecutive cosine similarities
            attn_cos = consecutive_cos_sim(capture.attn_outputs, all_layers)
            mlp_cos = consecutive_cos_sim(capture.mlp_outputs, all_layers)
            delta_cos = consecutive_cos_sim(deltas, all_layers)

            for k in range(N_LAYERS - 1):
                if attn_cos[k] is not None:
                    attn_cos_sums[k] += attn_cos[k]
                    attn_counts[k] += 1
                if mlp_cos[k] is not None:
                    mlp_cos_sums[k] += mlp_cos[k]
                    mlp_counts[k] += 1
                if delta_cos[k] is not None:
                    delta_cos_sums[k] += delta_cos[k]
                    delta_counts[k] += 1

            if (b_idx + 1) % 5 == 0:
                print(f"  batch {b_idx + 1}/{n_full_batches}")

    # Average
    attn_cos_avg = [s / c if c > 0 else None for s, c in zip(attn_cos_sums, attn_counts)]
    mlp_cos_avg = [s / c if c > 0 else None for s, c in zip(mlp_cos_sums, mlp_counts)]
    delta_cos_avg = [s / c if c > 0 else None for s, c in zip(delta_cos_sums, delta_counts)]

    # Print results
    print(f"\n{'=' * 70}")
    print("  Consecutive-layer cosine similarity (v2)")
    print(f"{'=' * 70}")
    print(f"{'Pair':>10s} {'Attn cos':>10s} {'MLP cos':>10s} {'Delta cos':>10s}")
    pair_labels = []
    for i in range(N_LAYERS - 1):
        label = f"L{i}-L{i+1}"
        pair_labels.append(label)
        a_str = f"{attn_cos_avg[i]:.4f}" if attn_cos_avg[i] is not None else "N/A"
        m_str = f"{mlp_cos_avg[i]:.4f}" if mlp_cos_avg[i] is not None else "N/A"
        d_str = f"{delta_cos_avg[i]:.4f}" if delta_cos_avg[i] is not None else "N/A"
        print(f"  {label:>8s} {a_str:>10s} {m_str:>10s} {d_str:>10s}")

    # Save to JSON for the figure script
    results = {
        "model": "G1_6S6E_deep_sparse_d768",
        "step": 2000,
        "n_layers": N_LAYERS,
        "energy_layers": ENERGY_LAYERS,
        "standard_layers": STANDARD_LAYERS,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "pair_labels": pair_labels,
        "attn_cos": attn_cos_avg,
        "mlp_cos": mlp_cos_avg,
        "delta_cos": delta_cos_avg,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    capture.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
