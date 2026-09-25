"""Measure output alignment across layers in the G1 6S6E model (v3).

v3 fixes TWO bugs that caused v2 to report zero energy-attention outputs:

1.  ROOT CAUSE: The model is loaded via meta-device + to_empty(), which zeros
    all non-persistent buffers. RoPE's cos_cached/sin_cached are non-persistent,
    so apply_rotary_pos_emb(Q, (zeros, zeros)) -> zero Q, zero K.
    FIX: call rope.reset_parameters() after loading to recompute from inv_freq.

2.  Defensive: strip _orig_mod wrappers from torch.compile (not needed here
    since the distcp checkpoint strips them, but harmless).

3.  For energy layers, reads block._attn_out directly after the forward pass
    instead of using a hook (avoids potential torch.compile wrapper issues).

Captures three signal types per layer:
  1. Attention output
  2. FF/MoE output
  3. Full layer delta (hidden_states_out - hidden_states_in)

Writes results to alignment_v3_results.json for the figure script.

Usage:
    bash experiments/boltzmann-moe/scripts/bsub/submit_gpu_test.sh measure_align_v3 \
        "python experiments/boltzmann-moe/scripts/measure_alignment_v3_20260925.py" 1 00:30 32G
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

OUT_JSON = REPO / "experiments/boltzmann-moe/scripts/alignment_v3_results.json"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_base: Path, device: str = "cuda") -> torch.nn.Module:
    import yaml
    torch._dynamo.reset()

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

    # ----- CRITICAL FIX: reinitialize RoPE cos/sin cache -----
    # The meta-device + to_empty() loading pattern zeroes all non-persistent
    # buffers. RoPE's cos_cached/sin_cached are non-persistent, so they end up
    # as all-zeros after loading. This makes apply_rotary_pos_emb return zero
    # for all Q/K, killing the energy attention output.
    rope = getattr(model.transformer, 'rope', None)
    if rope is not None:
        print("  Reinitializing RoPE cos/sin cache (non-persistent buffers were zeroed by meta device load)")
        # Check current state
        cos_before = rope.cos_cached.float().norm().item() if hasattr(rope, 'cos_cached') else -1
        rope.reset_parameters()
        # Move cache to same device/dtype
        rope.cos_cached = rope.cos_cached.to(device=device, dtype=torch.bfloat16)
        rope.sin_cached = rope.sin_cached.to(device=device, dtype=torch.bfloat16)
        cos_after = rope.cos_cached.float().norm().item()
        print(f"    cos_cached norm: {cos_before:.4f} -> {cos_after:.4f}")
    # ----- END FIX -----

    print(f"  Model loaded on {device}, dtype=bf16")
    return model, step


# ---------------------------------------------------------------------------
# Capture: hooks for standard layers; _attn_out for energy layers
# ---------------------------------------------------------------------------

class OutputCaptureV3:
    """Capture attention, MLP, and full-layer-delta outputs for all layers.

    Standard layers (0-5): hooks on sequence_mixer and mlp_block.
    Energy layers (6-11): reads block._attn_out after forward (no hook needed).
                          Hook on ffwd for MLP output only.
    All layers: block-level hooks for input/output to compute deltas.
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

            # Block-level hook: capture input and output for delta
            self._hooks.append(
                block.register_forward_hook(self._make_block_hook(i))
            )

            if i in ENERGY_LAYERS:
                # Energy block: hook on ffwd for MLP output
                self._hooks.append(
                    block.ffwd.register_forward_hook(self._make_hook(self.mlp_outputs, i))
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
    def _make_hook(store: dict, layer_idx: int):
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                output = output[0]
            store[layer_idx] = output.detach()
        return hook

    def _make_block_hook(self, layer_idx: int):
        def hook(_module, input_args, output):
            inp = input_args[0] if isinstance(input_args, tuple) else input_args
            if isinstance(inp, torch.Tensor):
                self.block_inputs[layer_idx] = inp.detach()
            if isinstance(output, tuple):
                output = output[0]
            self.block_outputs[layer_idx] = output.detach()
        return hook

    def read_energy_attn_from_blocks(self):
        """Read _attn_out from energy blocks after forward pass."""
        for i in ENERGY_LAYERS:
            block = self._model.transformer.h[i]
            attn_out = getattr(block, '_attn_out', None)
            if attn_out is not None:
                self.attn_outputs[i] = attn_out.detach()
            else:
                print(f"  [L{i}] WARNING: _attn_out not found!")

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


def compute_deltas(inputs, outputs):
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
    print("  G1 6S6E alignment measurement (v3)")
    print("  Fix: reinitialize RoPE cache after meta-device load")
    print("=" * 60)

    model, step = load_model(CKPT_BASE, device=DEVICE)
    capture = OutputCaptureV3(model)

    all_layers = list(range(N_LAYERS))
    batch_size = 4
    n_full_batches = N_SEQS // batch_size

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
            capture.read_energy_attn_from_blocks()

            # Diagnostics on first batch
            if b_idx == 0:
                print("\n--- Diagnostics (batch 0) ---")
                for i in range(N_LAYERS):
                    attn_v = capture.attn_outputs.get(i)
                    mlp_v = capture.mlp_outputs.get(i)
                    tag = "E" if i in ENERGY_LAYERS else "S"
                    attn_info = f"norm={attn_v.float().norm().item():.4f}" if attn_v is not None else "MISSING"
                    mlp_info = f"norm={mlp_v.float().norm().item():.4f}" if mlp_v is not None else "MISSING"
                    print(f"  L{i:2d} [{tag}]: attn={attn_info:>16s}  mlp={mlp_info:>16s}")

            # Compute deltas and cosine similarities
            deltas = compute_deltas(capture.block_inputs, capture.block_outputs)
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
    print(f"  Consecutive-layer cosine similarity (v3, step {step})")
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

    results = {
        "model": "G1_6S6E_deep_sparse_d768",
        "step": step,
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
