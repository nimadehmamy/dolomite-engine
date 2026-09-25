"""Measure output and weight alignment across layers in the G1 6S6E model.

Loads the G1 checkpoint (6 softmax + 6 Boltzmann layers, d=768) from distcp,
runs ~100 random-token sequences through it, and computes:

  1. 12x12 cosine-similarity matrices for attention and MLP *outputs*
     (mean over tokens, averaged across sequences).
  2. 6x6 weight alignment for the Boltzmann MoE layers (6-11):
     cosine similarity of fused W1 and W2 expert matrices between layer pairs.

Usage (submit via the GPU test script -- do NOT run directly):
    bash experiments/boltzmann-moe/scripts/bsub/submit_gpu_test.sh measure_align \
        "python experiments/boltzmann-moe/scripts/measure_alignment_20260925.py" 1 01:00 32G
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Register custom model types (energy, register_energy, etc.)
import lm_engine.hf_models  # noqa: F401

CKPT_BASE = REPO / "experiments/boltzmann-moe/results/fwe/fineweb_G1_6S6E_deep_sparse_d768"
N_LAYERS = 12
ENERGY_LAYERS = list(range(6, 12))  # layers 6-11 are Boltzmann MoE
STANDARD_LAYERS = list(range(0, 6))  # layers 0-5 are softmax + Switch MoE

# Forward-pass settings
N_SEQS = 100
SEQ_LEN = 512
VOCAB_SIZE = 100352
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_base: Path, device: str = "cuda") -> torch.nn.Module:
    """Load the model from a distributed checkpoint (single-GPU, no FSDP)."""
    import yaml

    iter_file = ckpt_base / "latest_checkpointed_iteration.json"
    step = json.load(open(iter_file))["latest_checkpointed_iteration"]
    print(f"Loading checkpoint at step {step}")

    ckpt_dir = ckpt_base / f"global_step{step}"
    model_dir = ckpt_dir / "model"
    config_path = ckpt_dir / "training_config.yml"

    with open(config_path) as f:
        train_cfg = yaml.safe_load(f)

    # Build model from the saved pretrained_config
    from transformers import AutoConfig
    config = AutoConfig.for_model(**train_cfg["model_args"]["pretrained_config"])

    from lm_engine.hf_models.models.energy.main import EnergyForCausalLM

    print("  Instantiating model on meta device ...")
    with torch.device("meta"):
        model = EnergyForCausalLM(config)
    model = model.to_empty(device="cpu")

    # Load distcp state dict
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

    # The checkpoint was saved by _ModelSaver which wraps EnergyForCausalLM
    # inside a ModelWrapper.  get_model_state_dict() on the wrapper produces
    # keys with a "model." prefix (wrapper.model -> CausalLM).  Loading
    # directly into EnergyForCausalLM requires stripping that prefix.
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
# Hook-based output capture
# ---------------------------------------------------------------------------

class OutputCapture:
    """Register forward hooks on attention and MLP submodules of each layer."""

    def __init__(self, model: torch.nn.Module):
        self.attn_outputs: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._register(model)

    def _register(self, model: torch.nn.Module):
        for i in range(N_LAYERS):
            block = model.transformer.h[i]
            if i in ENERGY_LAYERS:
                # Energy block: block.attn, block.ffwd
                self._hooks.append(
                    block.attn.register_forward_hook(self._make_hook(self.attn_outputs, i))
                )
                self._hooks.append(
                    block.ffwd.register_forward_hook(self._make_hook(self.mlp_outputs, i))
                )
            else:
                # Standard GPT block: block.sequence_mixer, block.mlp_block
                self._hooks.append(
                    block.sequence_mixer.register_forward_hook(self._make_hook(self.attn_outputs, i))
                )
                self._hooks.append(
                    block.mlp_block.register_forward_hook(self._make_hook(self.mlp_outputs, i))
                )

    @staticmethod
    def _make_hook(store: dict, layer_idx: int):
        def hook(_module, _input, output):
            # Some modules return tuples; take the first element
            if isinstance(output, tuple):
                output = output[0]
            store[layer_idx] = output.detach()
        return hook

    def clear(self):
        self.attn_outputs.clear()
        self.mlp_outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def pairwise_cos_sim(vecs: dict[int, torch.Tensor], layers: list[int]) -> torch.Tensor:
    """Compute NxN matrix of mean per-token cosine similarity between layer outputs.

    Each vecs[i] has shape (B, T, D). We flatten to (B*T, D), normalise, and
    compute the mean cosine similarity for every pair (i, j).
    """
    n = len(layers)
    mat = torch.zeros(n, n)
    flat = {}
    for idx, li in enumerate(layers):
        v = vecs[li].float().reshape(-1, vecs[li].shape[-1])  # (B*T, D)
        flat[idx] = F.normalize(v, dim=-1)

    for a in range(n):
        for b in range(a, n):
            cos = (flat[a] * flat[b]).sum(-1).mean().item()
            mat[a, b] = cos
            mat[b, a] = cos
    return mat


def print_matrix(mat: torch.Tensor, labels: list[str], title: str):
    """Pretty-print a square matrix."""
    n = mat.shape[0]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    # Header
    hdr = "        " + "".join(f"{l:>8s}" for l in labels)
    print(hdr)
    for i in range(n):
        row = f"{labels[i]:>8s}" + "".join(f"{mat[i, j]:8.3f}" for j in range(n))
        print(row)


# ---------------------------------------------------------------------------
# Weight alignment (Boltzmann MoE layers only)
# ---------------------------------------------------------------------------

def measure_weight_alignment(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 6x6 cosine similarity of fused W1 and W2 between energy layers.

    For each pair (i, j) of energy layers, we compute the mean cosine similarity
    across the K=8 experts: for expert k, cos(W_i[k].flatten(), W_j[k].flatten()).
    Returns (w1_sim, w2_sim) each of shape (6, 6).
    """
    n = len(ENERGY_LAYERS)
    w1_sim = torch.zeros(n, n)
    w2_sim = torch.zeros(n, n)

    # Collect W1 and W2 for each energy layer, viewed as (K, I_e, H)
    w1_list, w2_list = [], []
    for li in ENERGY_LAYERS:
        block = model.transformer.h[li]
        holder = block.ffwd.expert_holder
        K = holder.n_experts
        I_e = holder.expert_I
        H = holder.hidden_size
        w1 = holder.W1.weight.data.float().view(K, I_e, H)  # (8, 431, 768)
        w2 = holder.W2.weight.data.float().view(K, I_e, H)
        w1_list.append(w1)
        w2_list.append(w2)

    for a in range(n):
        for b in range(a, n):
            # Mean cosine similarity across experts
            cos_w1 = 0.0
            cos_w2 = 0.0
            K = w1_list[a].shape[0]
            for k in range(K):
                cos_w1 += F.cosine_similarity(
                    w1_list[a][k].flatten().unsqueeze(0),
                    w1_list[b][k].flatten().unsqueeze(0),
                ).item()
                cos_w2 += F.cosine_similarity(
                    w2_list[a][k].flatten().unsqueeze(0),
                    w2_list[b][k].flatten().unsqueeze(0),
                ).item()
            cos_w1 /= K
            cos_w2 /= K
            w1_sim[a, b] = cos_w1
            w1_sim[b, a] = cos_w1
            w2_sim[a, b] = cos_w2
            w2_sim[b, a] = cos_w2

    return w1_sim, w2_sim


def measure_weight_alignment_subsampled(
    model: torch.nn.Module, n_rows: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as above but using random row subsampling for speed.

    Randomly selects n_rows out of I_e rows per expert and computes cosine similarity
    on those. Changes the random indices each call for coverage.
    """
    n = len(ENERGY_LAYERS)
    w1_sim = torch.zeros(n, n)
    w2_sim = torch.zeros(n, n)

    w1_list, w2_list = [], []
    for li in ENERGY_LAYERS:
        block = model.transformer.h[li]
        holder = block.ffwd.expert_holder
        K = holder.n_experts
        I_e = holder.expert_I
        H = holder.hidden_size
        w1 = holder.W1.weight.data.float().view(K, I_e, H)
        w2 = holder.W2.weight.data.float().view(K, I_e, H)
        w1_list.append(w1)
        w2_list.append(w2)

    I_e = w1_list[0].shape[1]
    K = w1_list[0].shape[0]
    sample = min(n_rows, I_e)
    idx = torch.randperm(I_e)[:sample]

    for a in range(n):
        for b in range(a, n):
            cos_w1 = 0.0
            cos_w2 = 0.0
            for k in range(K):
                v1a = w1_list[a][k, idx].flatten().unsqueeze(0)
                v1b = w1_list[b][k, idx].flatten().unsqueeze(0)
                v2a = w2_list[a][k, idx].flatten().unsqueeze(0)
                v2b = w2_list[b][k, idx].flatten().unsqueeze(0)
                cos_w1 += F.cosine_similarity(v1a, v1b).item()
                cos_w2 += F.cosine_similarity(v2a, v2b).item()
            cos_w1 /= K
            cos_w2 /= K
            w1_sim[a, b] = cos_w1
            w1_sim[b, a] = cos_w1
            w2_sim[a, b] = cos_w2
            w2_sim[b, a] = cos_w2

    return w1_sim, w2_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  G1 6S6E alignment measurement")
    print("=" * 60)

    model = load_model(CKPT_BASE, device=DEVICE)
    capture = OutputCapture(model)

    # Accumulators for cosine similarity across batches
    attn_cos_sum = torch.zeros(N_LAYERS, N_LAYERS)
    mlp_cos_sum = torch.zeros(N_LAYERS, N_LAYERS)
    n_batches = 0
    all_layers = list(range(N_LAYERS))

    print(f"\nRunning {N_SEQS} sequences of length {SEQ_LEN} ...")
    batch_size = 4
    n_full_batches = N_SEQS // batch_size

    with torch.no_grad():
        for b_idx in range(n_full_batches):
            # Random token IDs
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
            capture.clear()

            # Forward pass (no labels, just get activations)
            model(input_ids=input_ids)

            # Compute pairwise cosine similarity for this batch
            attn_cos = pairwise_cos_sim(capture.attn_outputs, all_layers)
            mlp_cos = pairwise_cos_sim(capture.mlp_outputs, all_layers)

            attn_cos_sum += attn_cos
            mlp_cos_sum += mlp_cos
            n_batches += 1

            if (b_idx + 1) % 5 == 0:
                print(f"  batch {b_idx + 1}/{n_full_batches}")

    # Average
    attn_cos_avg = attn_cos_sum / n_batches
    mlp_cos_avg = mlp_cos_sum / n_batches

    labels = [f"L{i}" for i in range(N_LAYERS)]
    print_matrix(attn_cos_avg, labels, "ATTENTION OUTPUT cosine similarity (12x12)")
    print_matrix(mlp_cos_avg, labels, "MLP/MoE OUTPUT cosine similarity (12x12)")

    # Consecutive-layer summary
    print(f"\n{'=' * 60}")
    print("  Consecutive-layer cosine similarity")
    print(f"{'=' * 60}")
    print(f"{'Pair':>10s} {'Attn cos':>10s} {'MLP cos':>10s}")
    for i in range(N_LAYERS - 1):
        print(f"  L{i}-L{i+1}    {attn_cos_avg[i, i+1]:10.4f} {mlp_cos_avg[i, i+1]:10.4f}")

    # Diagonal (self-similarity, should be 1.0)
    print(f"\n  Diagonal check (should be ~1.0):")
    print(f"  Attn diag: {[f'{attn_cos_avg[i,i]:.3f}' for i in range(N_LAYERS)]}")
    print(f"  MLP  diag: {[f'{mlp_cos_avg[i,i]:.3f}' for i in range(N_LAYERS)]}")

    # --- Weight alignment for Boltzmann MoE layers (6-11) ---
    print(f"\n{'=' * 60}")
    print("  WEIGHT ALIGNMENT (Boltzmann MoE layers 6-11)")
    print(f"{'=' * 60}")

    w1_sim, w2_sim = measure_weight_alignment(model)
    e_labels = [f"L{i}" for i in ENERGY_LAYERS]
    print_matrix(w1_sim, e_labels, "W1 weight cosine similarity (6x6, mean over 8 experts)")
    print_matrix(w2_sim, e_labels, "W2 weight cosine similarity (6x6, mean over 8 experts)")

    # Subsampled version for comparison
    w1_sub, w2_sub = measure_weight_alignment_subsampled(model, n_rows=64)
    print_matrix(w1_sub, e_labels, "W1 weight cos sim (subsampled 64 rows, 6x6)")
    print_matrix(w2_sub, e_labels, "W2 weight cos sim (subsampled 64 rows, 6x6)")

    # Also print the PSDA projection matrices' alignment (these are per-layer d x d)
    print(f"\n{'=' * 60}")
    print("  PSDA Projection alignment (S matrices, layers 6-11)")
    print(f"{'=' * 60}")
    n_e = len(ENERGY_LAYERS)
    proj_sim = torch.zeros(n_e, n_e)
    S_list = []
    for li in ENERGY_LAYERS:
        block = model.transformer.h[li]
        S = block.proj.S.data.float()  # (768, rank)
        S_list.append(S)
    for a in range(n_e):
        for b in range(a, n_e):
            cos = F.cosine_similarity(
                S_list[a].flatten().unsqueeze(0),
                S_list[b].flatten().unsqueeze(0),
            ).item()
            proj_sim[a, b] = cos
            proj_sim[b, a] = cos
    print_matrix(proj_sim, e_labels, "PSDA S-matrix cosine similarity (6x6)")

    # scale_ff values
    print(f"\n  scale_ff values per energy layer:")
    for li in ENERGY_LAYERS:
        sf = model.transformer.h[li].scale_ff.item()
        print(f"    L{li}: scale_ff = {sf:.4f}")

    capture.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
