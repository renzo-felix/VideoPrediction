"""
direction_vectors_layerwise.py
==============================
Compute speed direction vectors at EVERY layer (0 to depth-1) in one shot.

El vector de dirección en la capa L es:
    d_L = normalize( mean(acts_fast_L) - mean(acts_slow_L) )

Por construcción apunta de "lento" hacia "rápido". Lo verificamos con
Pearson r: si r > 0, la dirección es correcta. Si r < 0, invertimos el signo.

Se hace UN solo forward pass por video con hooks en TODAS las capas,
lo que evita repetir la inferencia 40 veces.

OUTPUTS:
  - results/direction_vectors_layerwise_{model}.pkl
  - figures/direction_vectors_layerwise_{model}.png  (3 subplots)

USO:
  python src/direction_vectors_layerwise.py --model vjepa
  python src/direction_vectors_layerwise.py --model videomae --checkpoint models/videomae_vitg.pth
"""

import sys
import argparse
import logging
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, ACTIVATIONS_DIR,
    VJEPA_CHECKPOINT, VIDEOMAE_CHECKPOINT,
    VJEPA_LAYER, VIDEOMAE_LAYER,
    IMG_SIZE, NUM_FRAMES, TUBELET_SIZE,
    STEERING_LOW_PERCENTILE, STEERING_HIGH_PERCENTILE,
)
from data import VideoDataset
from models import load_vjepa, load_videomae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-layer activation extractor
# ---------------------------------------------------------------------------

class MultiLayerExtractor:
    """
    Registra hooks en TODOS los bloques del modelo en un solo forward pass.
    Esto es mucho más eficiente que llamar al modelo 40 veces.
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.activations = {}   # {layer_idx: tensor [B, N, D]}
        self._hooks = []
        self.num_layers = len(model.blocks)

        for i in range(self.num_layers):
            h = model.blocks[i].register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def fn(module, input, output):
            # output: [B, N, D] — mean pool sobre tokens → [B, D]
            self.activations[layer_idx] = output.detach().cpu().mean(dim=1)
        return fn

    def __call__(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: [B, C, T, H, W]
        Returns:
            {layer_idx: np.array [B, D]}
        """
        self.activations = {}
        with torch.no_grad():
            if self.model_name == 'videomae':
                B = frames.shape[0]
                num_patches = self.model.patch_embed.num_patches
                mask = torch.zeros(B, num_patches, dtype=torch.bool, device=frames.device)
                self.model(frames, mask=mask)
            else:
                self.model(frames)

        return {k: v.numpy() for k, v in self.activations.items()}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Extracción de activaciones multi-capa para todos los videos
# ---------------------------------------------------------------------------

def extract_all_layers(model, model_name: str, metadata_path: Path,
                       videos_base_dir: Path, device: torch.device) -> tuple:
    """
    Extrae activaciones en todas las capas para todos los videos del CSV.

    Returns:
        acts_by_layer: {layer_idx: np.array [N_videos, D]}
        speeds: np.array [N_videos]  (velocidad real de cada video)
        df: DataFrame con metadata completa
    """
    df = pd.read_csv(metadata_path)
    dataset = VideoDataset(metadata_path, videos_base_dir, IMG_SIZE, NUM_FRAMES)

    extractor = MultiLayerExtractor(model, model_name)
    num_layers = extractor.num_layers

    # Pre-asignar
    embed_dim = None
    all_acts = None    # {layer_idx: list of [D]}
    speeds = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue

        frames = sample['frames'].unsqueeze(0).to(device)  # [1, C, T, H, W]
        layer_acts = extractor(frames)  # {layer_idx: [1, D]}

        if embed_dim is None:
            embed_dim = layer_acts[0].shape[1]
            all_acts = {i: [] for i in range(num_layers)}

        for i in range(num_layers):
            all_acts[i].append(layer_acts[i][0])  # [D]

        speeds.append(sample['actual_speed'])

        if (idx + 1) % 20 == 0:
            logger.info(f"  Processed {idx+1}/{len(dataset)} videos")

    extractor.remove()

    acts_by_layer = {i: np.stack(all_acts[i], axis=0) for i in range(num_layers)}
    speeds = np.array(speeds)

    logger.info(f"Extracted activations: {len(speeds)} videos, {num_layers} layers, dim={embed_dim}")
    return acts_by_layer, speeds, df


# ---------------------------------------------------------------------------
# Cálculo de direction vectors
# ---------------------------------------------------------------------------

def compute_direction_vectors(acts_by_layer: dict, speeds: np.ndarray,
                               percentile_low: float = 33,
                               percentile_high: float = 67) -> dict:
    """
    Para cada capa L calcula:
      - d_L: direction vector normalizado (apunta hacia "rápido")
      - pearson_r: correlación entre proyecciones y velocidad real
      - effect_size: Cohen's d entre grupos alto/bajo
      - low_mean, high_mean: medias de activación por grupo

    La dirección d_L = mean(fast) - mean(slow) apunta por construcción
    hacia activaciones de videos rápidos. Si pearson_r < 0 se invierte.
    """
    low_thresh = np.percentile(speeds, percentile_low)
    high_thresh = np.percentile(speeds, percentile_high)

    low_mask = speeds <= low_thresh
    high_mask = speeds >= high_thresh

    logger.info(f"Speed groups — low (≤{low_thresh:.1f}): {low_mask.sum()} videos, "
                f"high (≥{high_thresh:.1f}): {high_mask.sum()} videos")

    results = {}

    for layer_idx in sorted(acts_by_layer.keys()):
        acts = acts_by_layer[layer_idx]  # [N, D]

        low_mean = acts[low_mask].mean(axis=0)   # [D]
        high_mean = acts[high_mask].mean(axis=0)  # [D]

        d = high_mean - low_mean                  # apunta hacia "rápido"
        norm = np.linalg.norm(d)
        if norm < 1e-8:
            d_normalized = d
        else:
            d_normalized = d / norm

        projections = acts @ d_normalized         # [N]  (score de velocidad)
        r, p = pearsonr(projections, speeds)

        # Si la dirección está invertida la corregimos
        if r < 0:
            d_normalized = -d_normalized
            projections = -projections
            r = -r

        effect_size = (high_mean - low_mean).mean() / (acts.std() + 1e-8)

        results[layer_idx] = {
            'direction_vector': d_normalized,
            'pearson_r': float(r),
            'pearson_p': float(p),
            'effect_size': float(effect_size),
            'low_mean': low_mean,
            'high_mean': high_mean,
            'low_speed_mean': float(speeds[low_mask].mean()),
            'high_speed_mean': float(speeds[high_mask].mean()),
        }

    return results


def compute_cosine_similarities(direction_results: dict) -> list:
    """
    Cosine similarity entre direction vectors de capas consecutivas.
    Alta similitud → el concepto se mantiene estable entre capas.
    """
    layers = sorted(direction_results.keys())
    sims = []
    for i in range(len(layers) - 1):
        d_curr = direction_results[layers[i]]['direction_vector']
        d_next = direction_results[layers[i + 1]]['direction_vector']
        sim = float(np.dot(d_curr, d_next))   # ambos normalizados → cosine
        sims.append(sim)
    return sims


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(direction_results: dict, cosine_sims: list,
                 model_name: str, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no disponible, saltando gráficas")
        return

    layers = sorted(direction_results.keys())
    r_values = [direction_results[l]['pearson_r'] for l in layers]
    effect_sizes = [direction_results[l]['effect_size'] for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f'Direction Vectors por Capa — {model_name.upper()}', fontsize=14)

    # Plot 1: Pearson r vs layer
    axes[0].plot(layers, r_values, 'b-o', markersize=4, linewidth=1.5)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Pearson r (velocidad)')
    axes[0].set_title('Correlación entre proyección sobre d_L y velocidad real')
    axes[0].set_ylim(-0.1, 1.0)
    axes[0].grid(True, alpha=0.3)
    best_layer = layers[int(np.argmax(r_values))]
    axes[0].axvline(best_layer, color='red', linestyle=':', alpha=0.7,
                    label=f'Mejor capa: {best_layer} (r={max(r_values):.3f})')
    axes[0].legend(fontsize=8)

    # Plot 2: Cosine similarity entre capas consecutivas
    axes[1].plot(layers[:-1], cosine_sims, 'g-o', markersize=4, linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Cosine similarity (d_L, d_{L+1})')
    axes[1].set_title('Estabilidad del direction vector entre capas consecutivas')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Effect size vs layer
    axes[2].plot(layers, effect_sizes, 'r-o', markersize=4, linewidth=1.5)
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel("Effect size (Cohen's d aprox.)")
    axes[2].set_title('Tamaño del efecto de velocidad por capa')
    axes[2].set_xlabel('Índice de capa')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Figura guardada: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute direction vectors at every layer')
    parser.add_argument('--model', default='vjepa', choices=['vjepa', 'videomae'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--metadata', type=str, default=None,
                        help='CSV con metadata de videos (default: data/training_multi_attribute_metadata.csv)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--low-percentile', type=float, default=STEERING_LOW_PERCENTILE)
    parser.add_argument('--high-percentile', type=float, default=STEERING_HIGH_PERCENTILE)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Paths
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        VJEPA_CHECKPOINT if args.model == 'vjepa' else VIDEOMAE_CHECKPOINT
    )
    metadata_path = Path(args.metadata) if args.metadata else (
        DATA_DIR / 'training_multi_attribute_metadata.csv'
    )
    output_pkl = Path(args.output) if args.output else (
        RESULTS_DIR / f'direction_vectors_layerwise_{args.model}.pkl'
    )
    output_fig = FIGURES_DIR / f'direction_vectors_layerwise_{args.model}.png'

    logger.info(f"Model:      {args.model}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Metadata:   {metadata_path}")
    logger.info(f"Device:     {device}")

    # 1. Cargar modelo
    logger.info("Loading model...")
    loader = load_vjepa if args.model == 'vjepa' else load_videomae
    model = loader(checkpoint, device)

    # 2. Extraer activaciones en todas las capas (UN forward pass por video)
    logger.info("Extracting activations from all layers...")
    acts_by_layer, speeds, df = extract_all_layers(
        model, args.model, metadata_path, DATA_DIR, device
    )

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 3. Calcular direction vectors por capa
    logger.info("Computing direction vectors...")
    direction_results = compute_direction_vectors(
        acts_by_layer, speeds,
        percentile_low=args.low_percentile,
        percentile_high=args.high_percentile,
    )

    # 4. Cosine similarities entre capas consecutivas
    cosine_sims = compute_cosine_similarities(direction_results)

    # 5. Resumen
    logger.info("\n" + "=" * 55)
    logger.info(f"{'Layer':>6}  {'Pearson r':>10}  {'Effect size':>12}")
    logger.info("=" * 55)
    for layer_idx in sorted(direction_results.keys()):
        r = direction_results[layer_idx]['pearson_r']
        d = direction_results[layer_idx]['effect_size']
        logger.info(f"{layer_idx:>6}  {r:>10.4f}  {d:>12.4f}")

    best_layer = max(direction_results, key=lambda l: direction_results[l]['pearson_r'])
    logger.info(f"\nMejor capa: {best_layer} "
                f"(r={direction_results[best_layer]['pearson_r']:.4f})")

    mean_cosine = float(np.mean(cosine_sims))
    logger.info(f"Cosine similarity promedio entre capas consecutivas: {mean_cosine:.4f}")
    if mean_cosine > 0.7:
        logger.info("→ Direction vector MUY ESTABLE entre capas (concepto consistente)")
    elif mean_cosine > 0.3:
        logger.info("→ Direction vector MODERADAMENTE ESTABLE entre capas")
    else:
        logger.info("→ Direction vector INESTABLE — el concepto se recodifica entre capas")

    # 6. Guardar resultados
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'direction_results': direction_results,
        'cosine_similarities': cosine_sims,
        'model': args.model,
        'speeds': speeds,
        'low_percentile': args.low_percentile,
        'high_percentile': args.high_percentile,
        'num_videos': len(speeds),
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(save_data, f)
    logger.info(f"Resultados guardados: {output_pkl}")

    # 7. Gráficas
    plot_results(direction_results, cosine_sims, args.model, output_fig)


if __name__ == '__main__':
    main()
