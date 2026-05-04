"""
analyze_patching_results.py
============================
Carga los resultados del experimento de directional patching y exporta:
  1. CSV con métricas clave por capa
  2. Figura resumen general
  3. Heatmap capas × alphas  ← muestra el patrón completo de un vistazo
  4. Hero curve capa mejor   ← curva real vs null con error bars (para paper)
  5. Scatter r vs mono       ← relación entre "codifica" y "causa"

USO:
  python experiments/src/analyze_patching_results.py --model vjepa
"""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from config import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_results(model: str, results_dir: Path) -> dict:
    path = results_dir / f"directional_patching_{model}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Corre primero run_directional_patching.sh")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CSV resumen
# ---------------------------------------------------------------------------

def build_summary_df(data: dict) -> pd.DataFrame:
    patching = data["patching_results"]
    mono     = data["monotonicity_scores"]
    rnd_mono = data.get("null_random_monotonicity", {})
    pca_mono = data.get("null_pca_monotonicity", {})

    rows = []
    for layer in sorted(patching.keys()):
        d = patching[layer]
        alphas_sorted = sorted(d["delta_by_alpha"].keys())
        deltas = [d["delta_by_alpha"][a] for a in alphas_sorted]
        m_real = mono.get(layer, 0.0)
        m_rnd  = rnd_mono.get(layer, 0.0)
        m_pca  = pca_mono.get(layer, 0.0)

        rows.append({
            "layer":               layer,
            "pearson_r":           round(d["direction_pearson_r"], 4),
            "monotonicity_real":   round(m_real, 4),
            "monotonicity_random": round(m_rnd, 4),
            "monotonicity_pca":    round(m_pca, 4),
            "null_gap":            round(m_real - max(m_rnd, m_pca), 4),
            "delta_alpha_neg3":    round(deltas[0], 4),
            "delta_alpha_pos3":    round(deltas[-1], 4),
            "baseline_score":      round(d["baseline"], 4),
            "n_videos":            d["n_videos"],
            "causal_verdict": (
                "FUERTE"  if m_real >= 1.0 else
                "PARCIAL" if m_real >= 0.7 else
                "DÉBIL"
            ),
            "null_validated": (m_real - max(m_rnd, m_pca) > 0.2) if rnd_mono else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figura 1: resumen general (existente, mejorada)
# ---------------------------------------------------------------------------

def plot_full_summary(data: dict, model: str, output_path: Path):
    patching  = data["patching_results"]
    mono      = data["monotonicity_scores"]
    rnd_mono  = data.get("null_random_monotonicity", {})
    pca_mono  = data.get("null_pca_monotonicity", {})
    rnd_patch = data.get("null_random_results", {})
    pca_patch = data.get("null_pca_results", {})
    has_nulls = bool(rnd_mono)

    layers = sorted(patching.keys())
    alphas = sorted(patching[layers[0]]["delta_by_alpha"].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    best_layer = max(mono, key=mono.get)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Directional Patching — {model.upper()} ViT-L  |  "
        f"Mejor capa: L{best_layer} (r={patching[best_layer]['direction_pearson_r']:.3f})",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.3)

    # Subplot 1: delta vs alpha todas las capas
    ax1 = fig.add_subplot(gs[0, 0])
    for i, layer in enumerate(layers):
        d = patching[layer]
        deltas = [d["delta_by_alpha"][a] for a in alphas]
        stds   = [d["std_by_alpha"].get(a, 0) for a in alphas]
        ax1.plot(alphas, deltas, "-o", color=colors[i],
                 label=f"L{layer}", markersize=4, linewidth=1.5)
        ax1.fill_between(alphas,
                         [dv - s for dv, s in zip(deltas, stds)],
                         [dv + s for dv, s in zip(deltas, stds)],
                         color=colors[i], alpha=0.08)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(0, color="gray",  linewidth=0.6, linestyle=":")
    ax1.set_title("Δ score vs alpha — todas las capas")
    ax1.set_xlabel("Alpha"); ax1.set_ylabel("Δ score velocidad")
    ax1.legend(fontsize=6, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: r + mono por capa
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(layers)); w = 0.35
    ax2.bar(x - w/2, [patching[l]["direction_pearson_r"] for l in layers],
            width=w, color="steelblue", alpha=0.8, label="Pearson r")
    ax2.bar(x + w/2, [mono[l] for l in layers],
            width=w, color="darkorange", alpha=0.8, label="Monotonicidad")
    ax2.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="Umbral 0.7")
    ax2.set_xticks(x); ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
    ax2.set_ylim(0, 1.1); ax2.set_title("Pearson r vs Monotonicidad por capa")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    # Subplot 3: null comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if has_nulls:
        w3 = 0.25
        ax3.bar(x - w3, [mono[l] for l in layers],          width=w3, color="steelblue",      label="Real",   alpha=0.85)
        ax3.bar(x,      [rnd_mono.get(l,0) for l in layers], width=w3, color="lightcoral",     label="Random", alpha=0.85)
        ax3.bar(x + w3, [pca_mono.get(l,0) for l in layers], width=w3, color="mediumseagreen", label="PCA",    alpha=0.85)
        ax3.axhline(0.7, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
        ax3.set_ylim(0, 1.15); ax3.set_title("Null experiments — monotonicidad")
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "Sin null experiments", ha="center", va="center", transform=ax3.transAxes)

    # Subplot 4 (fila 2 completa): delta(α=+3) real vs nulls
    ax4 = fig.add_subplot(gs[1, :])
    delta3_real = [patching[l]["delta_by_alpha"].get(3.0, 0) for l in layers]
    if has_nulls and rnd_patch:
        w4 = 0.25
        delta3_rnd = [rnd_patch[l]["delta_by_alpha"].get(3.0, 0) for l in layers if l in rnd_patch]
        delta3_pca = [pca_patch[l]["delta_by_alpha"].get(3.0, 0) for l in layers if l in pca_patch]
        ax4.bar(x - w4, delta3_real, width=w4, color="steelblue",      alpha=0.85, label="Real δ(α=+3)")
        ax4.bar(x,      delta3_rnd,  width=w4, color="lightcoral",     alpha=0.75, label="Random δ(α=+3)")
        ax4.bar(x + w4, delta3_pca,  width=w4, color="mediumseagreen", alpha=0.75, label="PCA δ(α=+3)")
    else:
        ax4.bar(x, delta3_real, color="steelblue", alpha=0.85, label="Real δ(α=+3)")
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.set_xticks(x); ax4.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax4.set_title("Efecto causal: Δ score a alpha=+3  (real >> null → vector específico de velocidad)")
    ax4.set_ylabel("Δ score velocidad"); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, axis="y")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[1/4] Figura resumen: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figura 2: Heatmap capas × alphas
# ---------------------------------------------------------------------------

def plot_heatmap(data: dict, model: str, output_path: Path):
    patching = data["patching_results"]
    layers   = sorted(patching.keys())
    alphas   = sorted(patching[layers[0]]["delta_by_alpha"].keys())

    matrix = np.array([
        [patching[l]["delta_by_alpha"].get(a, 0) for a in alphas]
        for l in layers
    ])

    vmax = np.abs(matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=ax, label="Δ score velocidad")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a:+.0f}" for a in alphas], fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}  (r={patching[l]['direction_pearson_r']:.2f})" for l in layers], fontsize=9)

    # Anotar valores en celdas
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > vmax * 0.5 else "black")

    ax.set_xlabel("Alpha (intensidad de la intervención)", fontsize=11)
    ax.set_ylabel("Capa", fontsize=11)
    ax.set_title(
        f"Heatmap: Δ score velocidad por capa y alpha — {model.upper()} ViT-L\n"
        "Rojo = empuja hacia rápido  |  Azul = empuja hacia lento  |  "
        "Gradiente vertical → causalidad",
        fontsize=11
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[2/4] Heatmap: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figura 3: Hero curve — mejor capa, real vs random vs PCA
# ---------------------------------------------------------------------------

def plot_hero_curve(data: dict, model: str, output_path: Path):
    patching  = data["patching_results"]
    mono      = data["monotonicity_scores"]
    rnd_patch = data.get("null_random_results", {})
    pca_patch = data.get("null_pca_results", {})

    best_layer = max(mono, key=mono.get)
    alphas = sorted(patching[best_layer]["delta_by_alpha"].keys())

    def get_curve(results, layer):
        if not results or layer not in results:
            return None, None
        d = results[layer]
        deltas = [d["delta_by_alpha"].get(a, 0) for a in alphas]
        stds   = [d["std_by_alpha"].get(a, 0)   for a in alphas]
        return np.array(deltas), np.array(stds)

    real_d, real_s = get_curve(patching, best_layer)
    rnd_d,  rnd_s  = get_curve(rnd_patch, best_layer)
    pca_d,  pca_s  = get_curve(pca_patch, best_layer)

    fig, ax = plt.subplots(figsize=(8, 5))

    def plot_with_band(ax, alphas, deltas, stds, color, label, zorder=2):
        ax.plot(alphas, deltas, "-o", color=color, label=label,
                markersize=7, linewidth=2.5, zorder=zorder)
        if stds is not None:
            ax.fill_between(alphas, deltas - stds, deltas + stds,
                            color=color, alpha=0.15, zorder=zorder - 1)

    plot_with_band(ax, alphas, real_d, real_s, "steelblue",      f"Speed vector (real)", zorder=4)
    if rnd_d is not None:
        plot_with_band(ax, alphas, rnd_d, rnd_s, "lightcoral",      "Random vector (null)", zorder=3)
    if pca_d is not None:
        plot_with_band(ax, alphas, pca_d, pca_s, "mediumseagreen",  "PCA PC1 (null)",       zorder=3)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray",  linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Alpha (escala de la intervención)", fontsize=12)
    ax.set_ylabel("Δ score velocidad en capa final", fontsize=12)
    ax.set_title(
        f"Hero curve — Capa {best_layer}  |  {model.upper()} ViT-L\n"
        f"r={patching[best_layer]['direction_pearson_r']:.3f}  |  "
        f"monotonicidad real={mono[best_layer]:.2f}\n"
        "La curva real debe ser monótona y mucho mayor que los nulls",
        fontsize=11
    )
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas); ax.set_xticklabels([f"{a:+.0f}" for a in alphas])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[3/4] Hero curve (L{best_layer}): {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figura 4: Scatter r vs monotonicidad
# ---------------------------------------------------------------------------

def plot_scatter_r_vs_mono(data: dict, model: str, output_path: Path):
    patching = data["patching_results"]
    mono     = data["monotonicity_scores"]
    rnd_mono = data.get("null_random_monotonicity", {})
    pca_mono = data.get("null_pca_monotonicity", {})

    layers  = sorted(patching.keys())
    r_vals  = [patching[l]["direction_pearson_r"] for l in layers]
    m_vals  = [mono[l] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Null backgrounds
    if rnd_mono:
        r_rnd = [patching[l]["direction_pearson_r"] for l in layers]
        m_rnd = [rnd_mono.get(l, 0) for l in layers]
        m_pca = [pca_mono.get(l, 0) for l in layers]
        ax.scatter(r_rnd, m_rnd, color="lightcoral",     s=60, alpha=0.6, label="Random (null)", zorder=2)
        ax.scatter(r_rnd, m_pca, color="mediumseagreen", s=60, alpha=0.6, label="PCA (null)",    zorder=2)

    # Real — coloreado por capa (oscuro=temprana, claro=tardía)
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    sc = ax.scatter(r_vals, m_vals, c=range(len(layers)), cmap="viridis",
                    s=120, zorder=4, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Índice de capa")

    for i, layer in enumerate(layers):
        ax.annotate(f"L{layer}", (r_vals[i], m_vals[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Umbral mono=0.7")
    ax.set_xlabel("Pearson r (correlación dirección ↔ velocidad)", fontsize=12)
    ax.set_ylabel("Monotonicidad (evidencia causal)", fontsize=12)
    ax.set_title(
        f"¿Las capas que codifican velocidad también la causan?\n"
        f"{model.upper()} ViT-L — cada punto es una capa",
        fontsize=11
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[4/4] Scatter r vs mono: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vjepa", choices=["vjepa", "videomae"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir  = Path(args.output_dir)  if args.output_dir  else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCargando resultados: {results_dir}")
    data = load_results(args.model, results_dir)

    # 1. CSV
    df = build_summary_df(data)
    csv_path = results_dir / f"summary_{args.model}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")
    print(df.to_string(index=False))
    print()

    # 2-5. Figuras
    plot_full_summary(data, args.model,    output_dir / f"fig1_resumen_{args.model}.png")
    plot_heatmap(data, args.model,          output_dir / f"fig2_heatmap_{args.model}.png")
    plot_hero_curve(data, args.model,       output_dir / f"fig3_hero_curve_{args.model}.png")
    plot_scatter_r_vs_mono(data, args.model,output_dir / f"fig4_scatter_{args.model}.png")

    print(f"\nTodo guardado en {output_dir}/")


if __name__ == "__main__":
    main()
