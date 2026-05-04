"""
analyze_patching_results.py
============================
Carga los resultados del experimento de directional patching y exporta:
  1. CSV con métricas clave por capa (para Excel / pandas)
  2. Figura resumen: curvas delta vs alpha + comparación null

USO:
  python experiments/src/analyze_patching_results.py --model vjepa
  python experiments/src/analyze_patching_results.py --model vjepa --output-dir experiments/analysis
"""

import sys
import argparse
import pickle
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import RESULTS_DIR, FIGURES_DIR


def load_results(model: str, results_dir: Path):
    path = results_dir / f"directional_patching_{model}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Corre primero run_directional_patching.sh")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_summary_df(data: dict) -> pd.DataFrame:
    """DataFrame con una fila por capa y todas las métricas importantes."""
    patching  = data["patching_results"]
    mono      = data["monotonicity_scores"]
    rnd_mono  = data.get("null_random_monotonicity", {})
    pca_mono  = data.get("null_pca_monotonicity", {})

    rows = []
    for layer in sorted(patching.keys()):
        d = patching[layer]
        alphas_sorted = sorted(d["delta_by_alpha"].keys())
        deltas = [d["delta_by_alpha"][a] for a in alphas_sorted]

        rows.append({
            "layer":              layer,
            "pearson_r":          round(d["direction_pearson_r"], 4),
            "monotonicity_real":  round(mono.get(layer, 0.0), 4),
            "monotonicity_random":round(rnd_mono.get(layer, 0.0), 4),
            "monotonicity_pca":   round(pca_mono.get(layer, 0.0), 4),
            "delta_alpha_neg3":   round(deltas[0], 4),
            "delta_alpha_0":      0.0,
            "delta_alpha_pos3":   round(deltas[-1], 4),
            "baseline_score":     round(d["baseline"], 4),
            "n_videos":           d["n_videos"],
            "causal_verdict":     (
                "FUERTE"  if mono.get(layer, 0) >= 1.0 else
                "PARCIAL" if mono.get(layer, 0) >= 0.7 else
                "DÉBIL"
            ),
            "null_validated":     (
                mono.get(layer, 0) - max(
                    rnd_mono.get(layer, 0), pca_mono.get(layer, 0)
                ) > 0.2
            ) if rnd_mono else None,
        })

    return pd.DataFrame(rows)


def plot_full_summary(data: dict, model: str, output_path: Path):
    patching  = data["patching_results"]
    mono      = data["monotonicity_scores"]
    rnd_mono  = data.get("null_random_monotonicity", {})
    pca_mono  = data.get("null_pca_monotonicity", {})
    rnd_patch = data.get("null_random_results", {})
    pca_patch = data.get("null_pca_results", {})

    layers = sorted(patching.keys())
    alphas = sorted(patching[layers[0]]["delta_by_alpha"].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    has_nulls = bool(rnd_mono)
    n_cols = 3 if has_nulls else 2
    fig = plt.figure(figsize=(6 * n_cols, 10))
    fig.suptitle(f"Directional Patching — {model.upper()} ViT-L\n"
                 f"Mejor capa: {max(mono, key=mono.get)} "
                 f"(r={patching[max(mono, key=mono.get)]['direction_pearson_r']:.3f})",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, n_cols, hspace=0.35, wspace=0.3)

    # --- Subplot 1: delta vs alpha (vector real) ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i, layer in enumerate(layers):
        d = patching[layer]
        deltas = [d["delta_by_alpha"][a] for a in alphas]
        stds   = [d["std_by_alpha"].get(a, 0) for a in alphas]
        label  = f"L{layer} (r={d['direction_pearson_r']:.2f})"
        ax1.plot(alphas, deltas, "-o", color=colors[i], label=label, markersize=4)
        ax1.fill_between(alphas,
                         [d - s for d, s in zip(deltas, stds)],
                         [d + s for d, s in zip(deltas, stds)],
                         color=colors[i], alpha=0.1)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(0, color="gray",  linewidth=0.6, linestyle=":")
    ax1.set_title("Δ score vs alpha — vector real")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Δ score velocidad")
    ax1.legend(fontsize=6, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Pearson r + monotonicidad por capa ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(layers))
    w = 0.35
    r_vals   = [patching[l]["direction_pearson_r"] for l in layers]
    mono_vals= [mono[l] for l in layers]
    ax2.bar(x - w/2, r_vals,    width=w, color="steelblue",  alpha=0.8, label="Pearson r")
    ax2.bar(x + w/2, mono_vals, width=w, color="darkorange", alpha=0.8, label="Monotonicidad")
    ax2.axhline(0.7, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Pearson r vs Monotonicidad")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Subplot 3: Null comparison (si existe) ---
    if has_nulls:
        ax3 = fig.add_subplot(gs[0, 2])
        rnd_vals = [rnd_mono.get(l, 0) for l in layers]
        pca_vals = [pca_mono.get(l, 0) for l in layers]
        w3 = 0.25
        ax3.bar(x - w3,   mono_vals, width=w3, color="steelblue",     label="Real",   alpha=0.85)
        ax3.bar(x,        rnd_vals,  width=w3, color="lightcoral",    label="Random", alpha=0.85)
        ax3.bar(x + w3,   pca_vals,  width=w3, color="mediumseagreen",label="PCA",    alpha=0.85)
        ax3.axhline(0.7, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
        ax3.set_ylim(0, 1.15)
        ax3.set_title("Null experiments\n(real vs random vs PCA)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

    # --- Subplot 4: delta(α=+3) por capa — real vs nulls ---
    ax4 = fig.add_subplot(gs[1, :])
    delta3_real = [patching[l]["delta_by_alpha"].get(3.0, 0) for l in layers]
    ax4.bar(x - w/2 if has_nulls else x, delta3_real,
            width=0.3, color="steelblue", alpha=0.85, label="Real δ(α=+3)")
    if has_nulls and rnd_patch:
        delta3_rnd = [rnd_patch[l]["delta_by_alpha"].get(3.0, 0) for l in layers if l in rnd_patch]
        delta3_pca = [pca_patch[l]["delta_by_alpha"].get(3.0, 0) for l in layers if l in pca_patch]
        ax4.bar(x,       delta3_rnd, width=0.3, color="lightcoral",     alpha=0.75, label="Random δ(α=+3)")
        ax4.bar(x + w/2, delta3_pca, width=0.3, color="mediumseagreen", alpha=0.75, label="PCA δ(α=+3)")
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax4.set_title("Efecto causal por capa: Δ score a alpha=+3\n"
                  "(real >> null → el vector de velocidad es causalmente específico)")
    ax4.set_ylabel("Δ score velocidad")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vjepa", choices=["vjepa", "videomae"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir  = Path(args.output_dir)  if args.output_dir  else FIGURES_DIR

    print(f"Cargando resultados de {results_dir}...")
    data = load_results(args.model, results_dir)

    # 1. CSV resumen
    df = build_summary_df(data)
    csv_path = results_dir / f"summary_{args.model}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")
    print(df.to_string(index=False))

    # 2. Figura completa
    fig_path = output_dir / f"full_summary_{args.model}.png"
    plot_full_summary(data, args.model, fig_path)


if __name__ == "__main__":
    main()
