"""
directional_patching.py
=======================
Prueba de causalidad mediante Directional Patching con escala alpha.

DIFERENCIA CON causal_patching_experiment.py (reemplazo total):
  - Ese script: reemplaza TODA la activación de la capa L con la del video A
  - Este script: solo agrega alpha * d_L a la activación (perturbación quirúrgica)

EXPERIMENTO:
  Para cada capa L y cada alpha en [-3, -2, -1, 0, 1, 2, 3]:
    1. Tomar videos "lentos" del conjunto de test
    2. Registrar hook en block[L] que suma alpha * d_L al output
    3. Pasar el video por el modelo con el hook activo
    4. Extraer activaciones de la capa final (layer_final)
    5. Proyectar sobre el direction vector de la capa final
    6. Medir cambio respecto a baseline (alpha=0)

INTERPRETACIÓN:
  Si la curva "score change vs alpha" es MONÓTONAMENTE CRECIENTE
  para múltiples capas → el direction vector es CAUSALMENTE RELEVANTE.

  La monotonicidad demuestra que no es solo correlación:
  estamos interviniéndolo artificialmente y el efecto se propaga.

OUTPUTS:
  - results/directional_patching_{model}.pkl
  - figures/directional_patching_{model}.png

USO:
  # Primero entrenar los direction vectors:
  python src/direction_vectors_layerwise.py --model vjepa

  # Luego correr el patching:
  python src/directional_patching.py --model vjepa
  python src/directional_patching.py --model videomae --layers 4 8 16 24 32 38
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
from sklearn.decomposition import PCA

from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    VJEPA_CHECKPOINT, VIDEOMAE_CHECKPOINT,
    VJEPA_LAYER, VIDEOMAE_LAYER,
    IMG_SIZE, NUM_FRAMES,
    TEST_CATEGORIES,
)
from data import VideoDataset
from models import load_vjepa, load_videomae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Valores de alpha a barrer (en unidades de norma del vector)
DEFAULT_ALPHAS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_additive_hook(d_L: np.ndarray, alpha: float, device: torch.device):
    """
    Hook que suma alpha * d_L al output del bloque.

    d_L es el direction vector normalizado [D].
    El output del bloque es [B, N, D] → sumamos [1, 1, D] (broadcast).

    alpha > 0 → empuja hacia "rápido"
    alpha < 0 → empuja hacia "lento"
    alpha = 0 → sin intervención (baseline)
    """
    d_tensor = torch.tensor(d_L, dtype=torch.float32, device=device).view(1, 1, -1)

    def hook_fn(module, input, output):
        return output + alpha * d_tensor

    return hook_fn


def make_capture_hook(storage: dict, key: str):
    """Hook que captura la activación mean-pooled de un bloque."""
    def hook_fn(module, input, output):
        storage[key] = output.detach().cpu().mean(dim=1)  # [B, D]
    return hook_fn


# ---------------------------------------------------------------------------
# Experimento de patching para un video
# ---------------------------------------------------------------------------

def run_directional_patch_single(
    model,
    model_name: str,
    frames: torch.Tensor,
    direction_results: dict,
    patch_layer: int,
    final_layer: int,
    alphas: list,
    device: torch.device,
) -> dict:
    """
    Ejecuta el patching para un solo video en una capa específica.

    Args:
        frames:           [1, C, T, H, W] video preprocesado
        direction_results: output de direction_vectors_layerwise.py
        patch_layer:      capa donde inyectamos alpha * d_L
        final_layer:      capa de donde extraemos la representación final
        alphas:           lista de valores de alpha a barrer

    Returns:
        {alpha: final_layer_speed_score}
    """
    d_patch = direction_results[patch_layer]['direction_vector']   # [D]
    d_final = direction_results[final_layer]['direction_vector']    # [D]

    results_by_alpha = {}
    capture = {}

    final_hook_handle = model.blocks[final_layer].register_forward_hook(
        make_capture_hook(capture, 'final')
    )

    for alpha in alphas:
        capture.clear()

        if abs(alpha) < 1e-8:
            # Baseline: sin hook de patching
            patch_handle = None
        else:
            patch_handle = model.blocks[patch_layer].register_forward_hook(
                make_additive_hook(d_patch, alpha, device)
            )

        with torch.no_grad():
            if model_name == 'videomae':
                B = frames.shape[0]
                num_patches = model.patch_embed.num_patches
                mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)
                model(frames, mask=mask)
            else:
                model(frames)

        if patch_handle is not None:
            patch_handle.remove()

        final_act = capture['final'].numpy()   # [1, D]
        speed_score = float(final_act[0] @ d_final)
        results_by_alpha[alpha] = speed_score

    final_hook_handle.remove()
    return results_by_alpha


# ---------------------------------------------------------------------------
# Experimento completo: barrer capas y alphas
# ---------------------------------------------------------------------------

def run_directional_patching(
    model,
    model_name: str,
    direction_results: dict,
    metadata_path: Path,
    videos_base_dir: Path,
    patch_layers: list,
    final_layer: int,
    alphas: list,
    device: torch.device,
    speed_group: str = 'slow',
    n_videos: int = 30,
) -> dict:
    """
    Corre el experimento completo.

    Args:
        patch_layers: capas donde intervenir (default: subset de todas)
        final_layer:  capa desde la que leer el efecto (default: última)
        speed_group:  'slow' o 'fast' — qué videos intervenir
        n_videos:     cuántos videos usar (más = más estable pero más lento)

    Returns:
        {patch_layer: {'mean_by_alpha': {alpha: mean_score},
                       'std_by_alpha':  {alpha: std_score},
                       'baseline':      float,
                       'pearson_r':     float de la capa}}
    """
    df = pd.read_csv(metadata_path)
    dataset = VideoDataset(metadata_path, videos_base_dir, IMG_SIZE, NUM_FRAMES)

    # Separar lentos/rápidos
    speeds = df['actual_speed'].values
    low_thresh = np.percentile(speeds, 33)
    high_thresh = np.percentile(speeds, 67)

    if speed_group == 'slow':
        target_mask = speeds <= low_thresh
    else:
        target_mask = speeds >= high_thresh

    target_indices = np.where(target_mask)[0]
    np.random.shuffle(target_indices)
    target_indices = target_indices[:n_videos]

    logger.info(f"Usando {len(target_indices)} videos '{speed_group}' "
                f"(velocidad {'≤' if speed_group=='slow' else '≥'} "
                f"{low_thresh if speed_group=='slow' else high_thresh:.1f})")

    all_results = {}

    for patch_layer in patch_layers:
        logger.info(f"\nPatching en capa {patch_layer} (final en capa {final_layer})...")

        scores_per_alpha = {a: [] for a in alphas}

        for i, idx in enumerate(target_indices):
            sample = dataset[idx]
            if sample is None:
                continue

            frames = sample['frames'].unsqueeze(0).to(device)  # [1, C, T, H, W]

            video_scores = run_directional_patch_single(
                model, model_name, frames,
                direction_results, patch_layer, final_layer, alphas, device
            )

            for alpha, score in video_scores.items():
                scores_per_alpha[alpha].append(score)

            if (i + 1) % 10 == 0:
                logger.info(f"  Video {i+1}/{len(target_indices)}")

        # Agregar resultados
        mean_by_alpha = {a: float(np.mean(scores_per_alpha[a]))
                         for a in alphas if scores_per_alpha[a]}
        std_by_alpha = {a: float(np.std(scores_per_alpha[a]))
                        for a in alphas if scores_per_alpha[a]}
        baseline = mean_by_alpha.get(0.0, 0.0)

        all_results[patch_layer] = {
            'mean_by_alpha': mean_by_alpha,
            'std_by_alpha': std_by_alpha,
            'delta_by_alpha': {a: mean_by_alpha[a] - baseline for a in mean_by_alpha},
            'baseline': baseline,
            'direction_pearson_r': direction_results[patch_layer]['pearson_r'],
            'n_videos': len(target_indices),
        }

        # Diagnóstico rápido de monotonicidad
        deltas = [all_results[patch_layer]['delta_by_alpha'][a]
                  for a in sorted(mean_by_alpha.keys())]
        is_monotone = all(x <= y for x, y in zip(deltas, deltas[1:]))
        logger.info(f"  Capa {patch_layer}: baseline={baseline:.4f}, "
                    f"delta(α=+3)={deltas[-1]:.4f}, monótonamente creciente: {is_monotone}")

    return all_results


# ---------------------------------------------------------------------------
# Verificación de monotonicidad (métrica clave)
# ---------------------------------------------------------------------------

def compute_monotonicity(patching_results: dict) -> dict:
    """
    Para cada capa calcula un score de monotonicidad de la curva delta vs alpha.
    Rango [0, 1]: 1 = perfectamente monótona creciente.
    """
    mono_scores = {}
    for layer, data in patching_results.items():
        alphas_sorted = sorted(data['delta_by_alpha'].keys())
        deltas = [data['delta_by_alpha'][a] for a in alphas_sorted]
        # Contar pares consecutivos donde aumenta
        increasing = sum(1 for a, b in zip(deltas, deltas[1:]) if b > a)
        total_pairs = len(deltas) - 1
        mono_scores[layer] = increasing / total_pairs if total_pairs > 0 else 0.0
    return mono_scores


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_patching_results(patching_results: dict, model_name: str,
                          output_path: Path, alphas: list):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib no disponible, saltando gráficas")
        return

    layers = sorted(patching_results.keys())
    n_layers = len(layers)
    colors = cm.viridis(np.linspace(0, 1, n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Directional Patching — {model_name.upper()}\n'
                 f'Intervención: video lento + alpha × d_L → score final layer',
                 fontsize=12)

    # Plot 1: delta vs alpha por capa
    ax = axes[0]
    for i, layer in enumerate(layers):
        data = patching_results[layer]
        alphas_sorted = sorted(data['delta_by_alpha'].keys())
        deltas = [data['delta_by_alpha'][a] for a in alphas_sorted]
        stds = [data['std_by_alpha'].get(a, 0) for a in alphas_sorted]
        label = f"L{layer} (r={data['direction_pearson_r']:.2f})"
        ax.plot(alphas_sorted, deltas, '-o', color=colors[i], label=label,
                markersize=4, linewidth=1.5)
        ax.fill_between(alphas_sorted,
                        [d - s for d, s in zip(deltas, stds)],
                        [d + s for d, s in zip(deltas, stds)],
                        color=colors[i], alpha=0.1)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Alpha (escala del direction vector)')
    ax.set_ylabel('Cambio en score de velocidad (vs baseline)')
    ax.set_title('Cambio en representación de velocidad por alpha')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: monotonicity score y pearson_r por capa
    mono_scores = compute_monotonicity(patching_results)
    r_values = [patching_results[l]['direction_pearson_r'] for l in layers]
    mono_values = [mono_scores[l] for l in layers]

    ax2 = axes[1]
    ax2.bar([l - 0.2 for l in layers], r_values, width=0.35,
            alpha=0.7, color='steelblue', label='Pearson r (dirección)')
    ax2.bar([l + 0.2 for l in layers], mono_values, width=0.35,
            alpha=0.7, color='darkorange', label='Monotonicidad [0,1]')
    ax2.set_xlabel('Capa')
    ax2.set_ylabel('Score')
    ax2.set_title('Pearson r vs Monotonicidad\n(causalidad = curva monótonamente creciente)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Figura guardada: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Null experiments (control estadístico)
# ---------------------------------------------------------------------------

def collect_activations_at_layers(
    model,
    model_name: str,
    dataset,
    target_indices: np.ndarray,
    layers: list,
    device: torch.device,
) -> dict:
    """
    Un forward pass por video, capturando activaciones mean-pooled en las
    capas indicadas. Necesario para construir el null PCA.

    Returns:
        {layer_idx: np.array [N, D]}
    """
    storage = {}
    hooks = []
    for layer in layers:
        def make_hook(l):
            def fn(module, inp, out):
                storage.setdefault(l, []).append(out.detach().cpu().mean(dim=1)[0].numpy())
            return fn
        hooks.append(model.blocks[layer].register_forward_hook(make_hook(layer)))

    for idx in target_indices:
        sample = dataset[idx]
        if sample is None:
            continue
        frames = sample['frames'].unsqueeze(0).to(device)
        with torch.no_grad():
            if model_name == 'videomae':
                B = frames.shape[0]
                mask = torch.zeros(B, model.patch_embed.num_patches, dtype=torch.bool, device=device)
                model(frames, mask=mask)
            else:
                model(frames)

    for h in hooks:
        h.remove()

    return {layer: np.stack(storage[layer], axis=0) for layer in layers if layer in storage}


def make_null_direction_results(direction_results: dict, null_type: str,
                                acts_by_layer: dict = None, seed: int = 0) -> dict:
    """
    Crea direction_results nulos para comparación:
      - 'random': vector Gaussiano aleatorio normalizado
      - 'pca':    primer componente principal de las activaciones

    Misma estructura que direction_results real para que corra en
    run_directional_patching sin cambios.
    """
    rng = np.random.RandomState(seed)
    null_results = {}
    for layer, data in direction_results.items():
        D = data['direction_vector'].shape[0]
        if null_type == 'random':
            vec = rng.randn(D)
        elif null_type == 'pca' and acts_by_layer is not None and layer in acts_by_layer:
            pca = PCA(n_components=1)
            pca.fit(acts_by_layer[layer])
            vec = pca.components_[0]
        else:
            vec = rng.randn(D)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        null_results[layer] = {**data, 'direction_vector': vec, 'pearson_r': 0.0}
    return null_results


def plot_null_comparison(
    real_mono: dict, random_mono: dict, pca_mono: dict,
    model_name: str, output_path: Path
):
    """
    Barra comparativa de monotonicidad: real vs random vs PCA por capa.
    Indica visualmente en qué capas el efecto causal es específico del
    direction vector de velocidad (real >> random/pca).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no disponible, saltando gráfica de null comparison")
        return

    layers = sorted(real_mono.keys())
    x = np.arange(len(layers))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.5), 5))
    ax.bar(x - w, [real_mono[l] for l in layers],   width=w, label='Real (speed vector)',  color='steelblue')
    ax.bar(x,      [random_mono[l] for l in layers], width=w, label='Null — random vector', color='lightcoral', alpha=0.8)
    ax.bar(x + w,  [pca_mono[l] for l in layers],    width=w, label='Null — PCA PC1',       color='mediumseagreen', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers], rotation=45)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color='gray', linestyle='--', linewidth=0.8, label='Umbral causal (0.7)')
    ax.set_xlabel('Capa')
    ax.set_ylabel('Monotonicidad [0, 1]')
    ax.set_title(f'Null Experiments — {model_name.upper()}\n'
                 f'Monotonicidad real vs. vectores control')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Figura null comparison guardada: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Directional Patching causal experiment')
    parser.add_argument('--model', default='vjepa', choices=['vjepa', 'videomae'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--direction-vectors', type=str, default=None,
                        help='Path al .pkl de direction_vectors_layerwise.py')
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Capas a patchear (default: subset equiespaciado)')
    parser.add_argument('--final-layer', type=int, default=None,
                        help='Capa desde donde medir el efecto (default: última)')
    parser.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    parser.add_argument('--n-videos', type=int, default=30,
                        help='Número de videos lentos a intervenir')
    parser.add_argument('--speed-group', default='slow', choices=['slow', 'fast'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--null-experiments', action='store_true', default=False,
                        help='Correr null experiments (random + PCA) para control estadístico')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # Paths
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        VJEPA_CHECKPOINT if args.model == 'vjepa' else VIDEOMAE_CHECKPOINT
    )
    direction_vectors_path = Path(args.direction_vectors) if args.direction_vectors else (
        RESULTS_DIR / f'direction_vectors_layerwise_{args.model}.pkl'
    )
    metadata_path = Path(args.metadata) if args.metadata else (
        DATA_DIR / 'training_multi_attribute_metadata.csv'
    )
    output_pkl = Path(args.output) if args.output else (
        RESULTS_DIR / f'directional_patching_{args.model}.pkl'
    )
    output_fig = FIGURES_DIR / f'directional_patching_{args.model}.png'

    logger.info(f"Model:             {args.model}")
    logger.info(f"Direction vectors: {direction_vectors_path}")
    logger.info(f"Device:            {device}")
    logger.info(f"Alphas:            {args.alphas}")
    logger.info(f"N videos:          {args.n_videos}")

    # 1. Cargar direction vectors pre-calculados
    if not direction_vectors_path.exists():
        logger.error(f"No se encontró {direction_vectors_path}. "
                     f"Ejecuta primero: python src/direction_vectors_layerwise.py --model {args.model}")
        sys.exit(1)

    with open(direction_vectors_path, 'rb') as f:
        dv_data = pickle.load(f)

    direction_results = dv_data['direction_results']
    num_layers = len(direction_results)
    logger.info(f"Direction vectors cargados: {num_layers} capas")

    # 2. Seleccionar capas a patchear
    if args.layers:
        patch_layers = args.layers
    else:
        # Default: 8 capas equiespaciadas para balance velocidad/cobertura
        step = max(1, num_layers // 8)
        patch_layers = list(range(0, num_layers, step))
        if (num_layers - 1) not in patch_layers:
            patch_layers.append(num_layers - 1)

    final_layer = args.final_layer if args.final_layer is not None else (num_layers - 1)

    logger.info(f"Capas a patchear:  {patch_layers}")
    logger.info(f"Capa final:        {final_layer}")

    # 3. Cargar modelo
    logger.info("Loading model...")
    loader = load_vjepa if args.model == 'vjepa' else load_videomae
    model = loader(checkpoint, device)
    model.eval()

    # 4. Correr experimento de patching
    logger.info("Running directional patching experiment...")
    patching_results = run_directional_patching(
        model, args.model,
        direction_results, metadata_path, DATA_DIR,
        patch_layers, final_layer,
        args.alphas, device,
        speed_group=args.speed_group,
        n_videos=args.n_videos,
    )

    # 5. Análisis de monotonicidad
    mono_scores = compute_monotonicity(patching_results)
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Capa':>6}  {'Pearson r':>10}  {'Monotonicidad':>14}  {'Causalidad':>12}")
    logger.info("=" * 60)
    for layer in sorted(patching_results.keys()):
        r = patching_results[layer]['direction_pearson_r']
        m = mono_scores[layer]
        causal = "FUERTE" if m >= 1.0 else "PARCIAL" if m >= 0.7 else "DÉBIL"
        logger.info(f"{layer:>6}  {r:>10.4f}  {m:>14.4f}  {causal:>12}")

    # ¿Cuántas capas tienen causalidad fuerte?
    strong_layers = [l for l, m in mono_scores.items() if m >= 1.0]
    logger.info(f"\nCapas con monotonicidad perfecta (causalidad fuerte): {strong_layers}")

    # 6. Guardar resultados
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'patching_results': patching_results,
        'monotonicity_scores': mono_scores,
        'model': args.model,
        'patch_layers': patch_layers,
        'final_layer': final_layer,
        'alphas': args.alphas,
        'n_videos': args.n_videos,
        'speed_group': args.speed_group,
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(save_data, f)
    logger.info(f"\nResultados guardados: {output_pkl}")

    # 7. Gráficas
    plot_patching_results(patching_results, args.model, output_fig, args.alphas)

    # 8. Null experiments
    if args.null_experiments:
        logger.info("\n" + "=" * 60)
        logger.info("NULL EXPERIMENTS (control estadístico)")
        logger.info("=" * 60)

        # Recolectar activaciones en patch_layers para PCA
        logger.info("Recolectando activaciones para null PCA...")
        df = pd.read_csv(metadata_path)
        dataset = VideoDataset(metadata_path, DATA_DIR, IMG_SIZE, NUM_FRAMES)
        speeds_all = df['actual_speed'].values
        low_thresh = np.percentile(speeds_all, 33)
        high_thresh = np.percentile(speeds_all, 67)
        target_mask = speeds_all <= low_thresh
        target_indices = np.where(target_mask)[0]
        np.random.shuffle(target_indices)
        target_indices = target_indices[:args.n_videos]

        acts_by_layer = collect_activations_at_layers(
            model, args.model, dataset, target_indices, patch_layers, device
        )

        # Null random
        null_random_dr = make_null_direction_results(direction_results, 'random', seed=args.seed)
        logger.info("Corriendo patching con vector ALEATORIO...")
        random_results = run_directional_patching(
            model, args.model, null_random_dr, metadata_path, DATA_DIR,
            patch_layers, final_layer, args.alphas, device,
            speed_group=args.speed_group, n_videos=args.n_videos,
        )
        random_mono = compute_monotonicity(random_results)

        # Null PCA
        null_pca_dr = make_null_direction_results(direction_results, 'pca', acts_by_layer, seed=args.seed)
        logger.info("Corriendo patching con vector PCA PC1...")
        pca_results = run_directional_patching(
            model, args.model, null_pca_dr, metadata_path, DATA_DIR,
            patch_layers, final_layer, args.alphas, device,
            speed_group=args.speed_group, n_videos=args.n_videos,
        )
        pca_mono = compute_monotonicity(pca_results)

        # Comparación
        logger.info(f"\n{'Capa':>6}  {'Real':>8}  {'Random':>8}  {'PCA':>8}  {'Veredicto':>12}")
        logger.info("-" * 55)
        for layer in sorted(mono_scores.keys()):
            r = mono_scores[layer]
            rnd = random_mono.get(layer, 0.0)
            pca = pca_mono.get(layer, 0.0)
            # Causal si real supera nulls por margen significativo
            gap = r - max(rnd, pca)
            verdict = "CAUSAL" if gap > 0.2 and r >= 0.7 else "DÉBIL" if gap > 0 else "NO CAUSAL"
            logger.info(f"{layer:>6}  {r:>8.3f}  {rnd:>8.3f}  {pca:>8.3f}  {verdict:>12}")

        # Guardar null results en el pkl
        save_data['null_random_results'] = random_results
        save_data['null_random_monotonicity'] = random_mono
        save_data['null_pca_results'] = pca_results
        save_data['null_pca_monotonicity'] = pca_mono
        with open(output_pkl, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"\nNull results añadidos a: {output_pkl}")

        # Gráfica comparativa
        null_fig = FIGURES_DIR / f'null_comparison_{args.model}.png'
        plot_null_comparison(mono_scores, random_mono, pca_mono, args.model, null_fig)


if __name__ == '__main__':
    main()
