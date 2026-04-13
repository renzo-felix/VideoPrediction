"""
plot_results.py — Grafica accuracy por capa del speed probe.
Uso: python speed_probe/plot_results.py --json speed_probe/output/vitl_ssv2/results.json
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out",  default=None, help="Ruta del PNG. Default: mismo dir que el json")
    args = p.parse_args()

    with open(args.json) as f:
        results = json.load(f)

    # Organizar por capa y componente
    data = {"residual": {}, "attn": {}, "mlp": {}}
    for key, val in results.items():
        # key: "block_0_residual"
        parts = key.split("_")
        layer = int(parts[1])
        comp  = parts[2]
        data[comp][layer] = val["accuracy"] * 100  # → porcentaje

    layers = sorted(data["residual"].keys())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"residual": "#e63946", "mlp": "#2a9d8f", "attn": "#457b9d"}
    styles = {"residual": "-o", "mlp": "-s", "attn": "-^"}

    for comp in ["residual", "mlp", "attn"]:
        ys = [data[comp][l] for l in layers]
        ax.plot(layers, ys, styles[comp], color=colors[comp],
                label=comp, linewidth=2, markersize=6)

    ax.set_xlabel("Capa del encoder", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("V-JEPA 2 (ViT-L) — Speed Linear Probe por capa · SSv2", fontsize=13)
    ax.set_xticks(layers)
    ax.set_ylim(60, 100)
    ax.axhline(55.7, color="gray", linestyle="--", linewidth=1, label="baseline (mayoría)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    import os
    out = args.out or os.path.join(os.path.dirname(args.json), "layer_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"Plot guardado: {out}")

    # Resumen en texto
    print("\nResumen (residual stream):")
    for l in layers:
        print(f"  capa {l:>2}: {data['residual'][l]:.1f}%")
    best_l = max(layers, key=lambda l: data["residual"][l])
    print(f"\nMejor capa: {best_l} ({data['residual'][best_l]:.1f}%)")

if __name__ == "__main__":
    main()
