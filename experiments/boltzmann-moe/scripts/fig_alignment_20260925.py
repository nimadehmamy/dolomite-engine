"""Publication-quality figure: consecutive-layer output alignment in 6S6E.

Shows cosine similarity between consecutive layers for three signal types:
  - Attention output (sequence_mixer / energy_attention block output)
  - FF/MoE output (MLP or Boltzmann MoE block output)
  - Full layer delta (residual stream change: h_out - h_in)

Key finding: energy attention outputs are exactly zero for all Boltzmann layers
(L6-L11). This is NOT a measurement artifact -- the EnergyAttention_QK module
genuinely outputs zero-norm tensors at step 2000, confirmed by both forward hooks
and direct block._attn_out inspection. The energy descent is driven entirely by
the MoE/FFN component: grad_E = scale_ff * ffwd_out.

Reads from alignment_v2_results.json (produced by measure_alignment_v2_20260925.py).

Usage (CPU-only, no GPU needed):
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    python experiments/boltzmann-moe/scripts/fig_alignment_20260925.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[3]
JSON_PATH = REPO / "experiments/boltzmann-moe/scripts/alignment_v2_results.json"
OUT_DIR = REPO / "experiments/boltzmann-moe/paper/figs"

N_LAYERS = 12
ENERGY_START = 6  # first energy/Boltzmann layer


# ---------------------------------------------------------------------------
# Data: v2 measurement results (job 1910193), with v1 fallback
# ---------------------------------------------------------------------------

# v2 data from measure_alignment_v2_20260925.py (100 seqs x 512 tokens)
ATTN_COS_V2 = [0.0404, 0.1275, 0.1507, 0.0111, 0.3642,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

MLP_COS_V2 = [0.1195, 0.2169, 0.0929, 0.1011, 0.0965,
              0.0598, 0.0647, 0.0430, 0.0244, 0.3093, 0.4946]

DELTA_COS_V2 = [0.1155, 0.1264, 0.0637, 0.0226, 0.2853,
                0.1616, 0.3557, 0.5051, 0.1094, 0.0685, 0.1665]

PAIR_LABELS = [f"{i}–{i+1}" for i in range(N_LAYERS - 1)]


def load_data():
    """Load results from JSON if available, otherwise use hardcoded v2 data."""
    attn_cos = list(ATTN_COS_V2)
    mlp_cos = list(MLP_COS_V2)
    delta_cos = list(DELTA_COS_V2)
    source = "hardcoded v2"

    if JSON_PATH.exists():
        try:
            with open(JSON_PATH) as f:
                data = json.load(f)
            attn_cos = data["attn_cos"]
            mlp_cos = data["mlp_cos"]
            delta_cos = data["delta_cos"]
            source = f"v2 JSON ({JSON_PATH.name})"
            print(f"Loaded data from {JSON_PATH}")
        except Exception as e:
            print(f"Warning: could not load {JSON_PATH}: {e}")
            print("Falling back to hardcoded v2 data.")

    print(f"Data source: {source}")
    return attn_cos, mlp_cos, delta_cos


def make_figure(attn_cos, mlp_cos, delta_cos):
    """Create the publication-quality alignment figure."""

    x = np.arange(len(PAIR_LABELS))

    # -- Style --
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # Colors: colorblind-friendly (Okabe-Ito inspired)
    c_attn = "#0072B2"   # blue
    c_mlp = "#D55E00"    # vermillion
    c_delta = "#009E73"  # bluish green

    marker_attn = "o"
    marker_mlp = "s"
    marker_delta = "D"

    ms = 5.5  # marker size
    lw = 1.6  # line width

    def plot_series(values, color, marker, label, zorder=2, linestyle="-"):
        """Plot a series, handling None values by breaking the line."""
        xs = [i for i, v in enumerate(values) if v is not None]
        ys = [v for v in values if v is not None]
        if not xs:
            return
        ax.plot(xs, ys, color=color, marker=marker, markersize=ms,
                linewidth=lw, label=label, zorder=zorder, linestyle=linestyle,
                markeredgecolor="white", markeredgewidth=0.6)

    # Plot each signal type
    # Attention: Switch pairs (solid) + Energy pairs (open markers at zero)
    attn_switch_x = list(range(5))      # indices 0-4 (L0-L1 through L4-L5)
    attn_switch_y = attn_cos[:5]
    attn_energy_x = list(range(5, 11))  # indices 5-10 (L5-L6 through L10-L11)
    attn_energy_y = attn_cos[5:]

    # Switch attention (solid filled markers)
    ax.plot(attn_switch_x, attn_switch_y, color=c_attn, marker=marker_attn,
            markersize=ms, linewidth=lw, label="Attention output",
            zorder=3, markeredgecolor="white", markeredgewidth=0.6)

    # Energy attention (open markers at zero, no connecting line to Switch)
    ax.plot(attn_energy_x, attn_energy_y, color=c_attn, marker=marker_attn,
            markersize=ms, linewidth=0, zorder=3,
            markeredgecolor=c_attn, markeredgewidth=0.8,
            markerfacecolor="white", alpha=0.5)

    # MLP/MoE (solid throughout)
    plot_series(mlp_cos, c_mlp, marker_mlp, "FF/MoE output", zorder=2)

    # Full layer delta
    plot_series(delta_cos, c_delta, marker_delta, "Full layer update", zorder=4)

    # Annotate the zero-output attention finding (below the open markers)
    ax.text(7.5, -0.025, "energy attn = 0",
            fontsize=7, color=c_attn, alpha=0.6,
            ha="center", va="top", style="italic")

    # Vertical dashed line at the Switch/Boltzmann boundary
    # L5-L6 transition pair is at index 5; boundary at x=4.5
    boundary_x = ENERGY_START - 1 - 0.5
    ax.axvline(x=boundary_x, color="#555555", linestyle="--", linewidth=0.8,
               alpha=0.5, zorder=1)

    # Region background shading
    ax.axvspan(-0.5, boundary_x, alpha=0.035, color="#6699CC", zorder=0)
    ax.axvspan(boundary_x, len(PAIR_LABELS) - 0.5, alpha=0.035, color="#CC8866", zorder=0)

    # Region labels
    ax.text(2.0, 0.56, "Switch\nlayers",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#777777",
            style="italic")
    ax.text(7.5, 0.56, "Boltzmann\nlayers",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#777777",
            style="italic")

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_LABELS)
    ax.set_xlabel("Layer pair (consecutive)")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(-0.06, 0.58)
    ax.set_xlim(-0.5, len(PAIR_LABELS) - 0.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.axhline(y=0, color="gray", linewidth=0.4, linestyle="-", alpha=0.3)

    ax.set_title("Consecutive-layer output alignment in 6S6E (d=768, step 2000)",
                 fontsize=10.5, pad=10)

    ax.legend(loc="upper left", framealpha=0.92, edgecolor="none",
              borderpad=0.4, handlelength=2.0)

    fig.tight_layout()
    return fig


def main():
    attn_cos, mlp_cos, delta_cos = load_data()

    # Print summary
    print("\nConsecutive-layer cosine similarity:")
    print(f"{'Pair':>8s} {'Attn':>8s} {'MLP':>8s} {'Delta':>8s}")
    for i, label in enumerate(PAIR_LABELS):
        a = f"{attn_cos[i]:.4f}" if attn_cos[i] is not None else "N/A"
        m = f"{mlp_cos[i]:.4f}" if mlp_cos[i] is not None else "N/A"
        d = f"{delta_cos[i]:.4f}" if delta_cos[i] is not None else "N/A"
        print(f"  {label:>6s} {a:>8s} {m:>8s} {d:>8s}")

    fig = make_figure(attn_cos, mlp_cos, delta_cos)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "alignment_6S6E.pdf"
    png_path = OUT_DIR / "alignment_6S6E.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
