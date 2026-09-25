"""Publication-quality figure: consecutive-layer output alignment in 6S6E (v3).

Shows cosine similarity between consecutive layers for three signal types:
  - Attention output (sequence_mixer / energy_attention block output)
  - FF/MoE output (MLP or Boltzmann MoE block output)
  - Full layer delta (residual stream change: h_out - h_in)

v3 update: uses corrected energy attention measurements from v3 script
(reads block._attn_out directly, fixing the torch.compile hook issue).

Reads from alignment_v3_results.json (produced by measure_alignment_v3_20260925.py).

Usage (CPU-only, no GPU needed):
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    python experiments/boltzmann-moe/scripts/fig_alignment_v3_20260925.py
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
JSON_PATH = REPO / "experiments/boltzmann-moe/scripts/alignment_v3_results.json"
OUT_DIR = REPO / "experiments/boltzmann-moe/paper/figs"

N_LAYERS = 12
ENERGY_START = 6  # first energy/Boltzmann layer

PAIR_LABELS = [f"{i}–{i+1}" for i in range(N_LAYERS - 1)]


def load_data():
    """Load results from v3 JSON."""
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"v3 results not found at {JSON_PATH}. Run measure_alignment_v3_20260925.py first.")

    with open(JSON_PATH) as f:
        data = json.load(f)
    attn_cos = data["attn_cos"]
    mlp_cos = data["mlp_cos"]
    delta_cos = data["delta_cos"]
    step = data.get("step", "?")
    print(f"Loaded data from {JSON_PATH} (step {step})")
    return attn_cos, mlp_cos, delta_cos, step


def make_figure(attn_cos, mlp_cos, delta_cos, step):
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

    # Plot each signal type as continuous lines
    plot_series(attn_cos, c_attn, marker_attn, "Attention output", zorder=3)
    plot_series(mlp_cos, c_mlp, marker_mlp, "FF/MoE output", zorder=2)
    plot_series(delta_cos, c_delta, marker_delta, "Full layer update", zorder=4)

    # Vertical dashed line at the Switch/Boltzmann boundary
    boundary_x = ENERGY_START - 1 - 0.5
    ax.axvline(x=boundary_x, color="#555555", linestyle="--", linewidth=0.8,
               alpha=0.5, zorder=1)

    # Region background shading
    ax.axvspan(-0.5, boundary_x, alpha=0.035, color="#6699CC", zorder=0)
    ax.axvspan(boundary_x, len(PAIR_LABELS) - 0.5, alpha=0.035, color="#CC8866", zorder=0)

    # Region labels
    ax.text(2.0, 0.97, "Switch\nlayers",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#777777",
            style="italic")
    ax.text(7.5, 0.97, "Boltzmann\nlayers",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#777777",
            style="italic")

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_LABELS)
    ax.set_xlabel("Layer pair (consecutive)")
    ax.set_ylabel("Mean cosine similarity")

    # Compute dynamic y-axis range
    all_vals = [v for v in attn_cos + mlp_cos + delta_cos if v is not None]
    y_min = min(all_vals) if all_vals else -0.1
    y_max = max(all_vals) if all_vals else 0.6
    margin = (y_max - y_min) * 0.12
    ax.set_ylim(min(y_min - margin, -0.06), y_max + margin + 0.06)
    ax.set_xlim(-0.5, len(PAIR_LABELS) - 0.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.axhline(y=0, color="gray", linewidth=0.4, linestyle="-", alpha=0.3)

    ax.set_title(f"Consecutive-layer output alignment in 6S6E (d=768, step {step})",
                 fontsize=10.5, pad=10)

    ax.legend(loc="upper left", framealpha=0.92, edgecolor="none",
              borderpad=0.4, handlelength=2.0)

    fig.tight_layout()
    return fig


def main():
    attn_cos, mlp_cos, delta_cos, step = load_data()

    # Print summary
    print("\nConsecutive-layer cosine similarity:")
    print(f"{'Pair':>8s} {'Attn':>8s} {'MLP':>8s} {'Delta':>8s}")
    for i, label in enumerate(PAIR_LABELS):
        a = f"{attn_cos[i]:.4f}" if attn_cos[i] is not None else "N/A"
        m = f"{mlp_cos[i]:.4f}" if mlp_cos[i] is not None else "N/A"
        d = f"{delta_cos[i]:.4f}" if delta_cos[i] is not None else "N/A"
        print(f"  {label:>6s} {a:>8s} {m:>8s} {d:>8s}")

    fig = make_figure(attn_cos, mlp_cos, delta_cos, step)

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
