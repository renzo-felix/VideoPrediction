"""
visualize_layers.py
===================
Visualiza cómo los tokens de 1-2 videos evolucionan capa a capa en V-JEPA 2.

INTENCIÓN: herramienta de interpretabilidad mecanística para 1-2 videos.
           NO para todos los videos — sería prohibitivo en computo y RAM.

Qué hace:
  1. Carga 1-2 videos y los pasa por el encoder congelado
  2. Extrae activaciones a nivel de TOKEN (no mean-pooled) por capa
  3. Ajusta PCA global sobre todos los tokens de todas las capas
     (proyección consistente → puedes comparar capas directamente)
  4. Genera un MP4 animado: cada frame = una capa, puntos = tokens,
     color = posición temporal (qué "rodaja" de frames representa el token)

Estimado de memoria (ViT-L, 2 videos, todas las capas, residual):
  - GPU: solo el forward pass (~2GB activaciones temporales en GPU)
  - CPU: 24 capas × 2048 tokens × 1024 dims × 4 bytes ≈ 200 MB por video
  - PCA fit: subsample a 50k tokens → ~200 MB máx

Uso:
  python speed_probe/visualize_layers.py \\
      --checkpoint checkpoints/vitl.pt \\
      --model_name vit_large \\
      --img_size 256 \\
      --num_frames 16 \\
      --videos data/ssv2/videos/20bn-something-something-v2/34899.webm \\
               data/ssv2/videos/20bn-something-something-v2/12345.webm \\
      --labels "Lanzar objeto (rápido)" "Sostener objeto (lento)" \\
      --output_dir speed_probe/output/vis
"""

import argparse, gc, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # sin pantalla (cluster)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed_probe.activation_extractor import VJEPA2ActivationExtractor
from speed_probe.run_speed_probe import load_video, load_encoder


# ── Extracción de tokens por capa ─────────────────────────────────────────────

def extract_token_activations(model, extractor, video_tensor, device, component):
    """
    Un video → dict {layer_idx: np.array [N_tokens, D]} para el componente dado.
    Los hooks ya están registrados; solo hace el forward y filtra por componente.
    """
    x = video_tensor.unsqueeze(0).to(device)   # [1, C, T, H, W]
    with torch.no_grad():
        model(x)

    raw = extractor.get_activations()           # {name: Tensor [1, N, D]}
    layer_acts = {}
    for name, act in raw.items():
        # name: "block_{i}_{component}"
        parts = name.split("_")
        if parts[2] != component:
            continue
        layer_i = int(parts[1])
        layer_acts[layer_i] = act.squeeze(0).float().numpy()   # [N, D]

    extractor.clear_activations()
    del x
    return layer_acts   # {layer_idx: [N_tokens, D]}


# ── PCA global ────────────────────────────────────────────────────────────────

def fit_global_pca(all_layer_acts, n_components=2, max_fit_tokens=50_000, seed=42):
    """
    Ajusta PCA sobre tokens de TODAS las capas y TODOS los videos.
    Subsamplea a max_fit_tokens para no reventar RAM.

    Por qué PCA global: la proyección 2D es la misma para todas las capas,
    entonces se puede comparar visualmente dónde "mueven" los tokens de capa a capa.
    """
    from sklearn.decomposition import PCA

    all_tokens = []
    for layer_acts in all_layer_acts:
        for arr in layer_acts.values():
            all_tokens.append(arr)   # [N, D]
    X = np.concatenate(all_tokens, axis=0)   # [total_tokens, D]

    if len(X) > max_fit_tokens:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_fit_tokens, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    print(f"  PCA fit sobre {len(X_fit):,} tokens ({X_fit.shape[1]}D → {n_components}D)...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_fit)
    ev = pca.explained_variance_ratio_
    print(f"  Varianza explicada: " + ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(ev)))
    del X, X_fit
    return pca


def project_all(all_layer_acts, pca):
    """Proyecta tokens de cada video y capa al espacio 2D de la PCA."""
    projected = []
    for layer_acts in all_layer_acts:
        proj = {li: pca.transform(arr) for li, arr in layer_acts.items()}  # [N, 2]
        projected.append(proj)
    return projected


# ── Colores por posición temporal ────────────────────────────────────────────

def temporal_colors(n_tokens, num_frames, tubelet_size=2, n_spatial_patches=256):
    """
    Asigna color entero [0..num_temporal_patches-1] a cada token según su
    posición temporal en el tubo spatio-temporal de V-JEPA 2.

    ViT-L: 16 frames / tubelet=2 = 8 posiciones temporales, 256 parches espaciales
           → 2048 tokens: [t0×256, t1×256, ..., t7×256]
    """
    n_temp = num_frames // tubelet_size          # 8 para 16 frames
    expected = n_temp * n_spatial_patches        # 2048
    colors = np.repeat(np.arange(n_temp), n_spatial_patches)
    if len(colors) != n_tokens:
        # Ajuste si el cálculo difiere (pe. img_size diferente)
        colors = np.tile(np.linspace(0, n_temp - 1, n_tokens // n_temp, dtype=int), n_temp)
        colors = colors[:n_tokens]
    return colors


# ── Animación ─────────────────────────────────────────────────────────────────

def make_animation(projected_per_video, labels, num_frames, layers_sorted,
                   output_path, component, fps=3):
    """
    MP4 animado: cada frame del video = una capa del transformer.
    Subplots: uno por video (máx 2).
    Colores: posición temporal del token (amarillo=tardío, morado=temprano).
    """
    n_videos = len(projected_per_video)

    # Límites globales de los ejes (consistentes entre todas las capas)
    all_pts = np.concatenate(
        [pts for proj in projected_per_video for pts in proj.values()], axis=0)
    pad = 0.08
    xr = all_pts[:, 0].max() - all_pts[:, 0].min()
    yr = all_pts[:, 1].max() - all_pts[:, 1].min()
    xlim = (all_pts[:, 0].min() - pad*xr, all_pts[:, 0].max() + pad*xr)
    ylim = (all_pts[:, 1].min() - pad*yr, all_pts[:, 1].max() + pad*yr)

    # Colores por token (mismo para todas las capas porque los tokens no cambian de orden)
    n_tokens = projected_per_video[0][layers_sorted[0]].shape[0]
    colors_per_video = [temporal_colors(n_tokens, num_frames) for _ in range(n_videos)]
    n_temp = num_frames // 2    # tubelet_size=2

    fig_w = 6.5 * n_videos
    fig, axes = plt.subplots(1, n_videos, figsize=(fig_w, 5.5))
    if n_videos == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0e0e0e")

    scatters = []
    for ax, label, proj, colors in zip(axes, labels, projected_per_video, colors_per_video):
        ax.set_facecolor("#1a1a2e")
        pts0 = proj[layers_sorted[0]]
        sc = ax.scatter(pts0[:, 0], pts0[:, 1], c=colors, cmap="plasma",
                        alpha=0.55, s=6, vmin=0, vmax=n_temp - 1,
                        linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.set_xlabel("PC1", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("PC2", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Posición temporal", color="#aaaaaa", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="#888888", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888888")
        scatters.append(sc)

    title = fig.suptitle(
        f"V-JEPA 2 — Token clustering · componente: {component} · capa {layers_sorted[0]}/{layers_sorted[-1]}",
        color="white", fontsize=11, y=0.97
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame_idx):
        layer = layers_sorted[frame_idx]
        for sc, proj in zip(scatters, projected_per_video):
            pts = proj[layer]
            sc.set_offsets(pts)
        title.set_text(
            f"V-JEPA 2 — Token clustering · componente: {component} · capa {layer}/{layers_sorted[-1]}"
        )
        return scatters + [title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(layers_sorted),
        interval=int(1000 / fps), blit=False
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                     extra_args=["-pix_fmt", "yuv420p"])
    ani.save(output_path, writer=writer, dpi=130)
    plt.close(fig)
    print(f"MP4 guardado: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Visualiza token clustering por capa de V-JEPA 2 (1-2 videos)."
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Ruta al encoder vitl.pt congelado")
    p.add_argument("--model_name",  required=True,
                   choices=["vit_large", "vit_huge", "vit_giant"])
    p.add_argument("--img_size",    type=int, required=True)
    p.add_argument("--num_frames",  type=int, required=True)
    p.add_argument("--videos",      required=True, nargs="+",
                   help="1-2 rutas a .webm/.mp4. Se recomiendan 2: uno rápido, uno lento.")
    p.add_argument("--labels",      nargs="+", default=None,
                   help="Nombre descriptivo de cada video (ej. 'Lanzar (rápido)')")
    p.add_argument("--component",   default="residual",
                   choices=["residual", "attn", "mlp"],
                   help="Componente a visualizar (default: residual)")
    p.add_argument("--all_layers",  action="store_true",
                   help="Usar TODAS las capas (más suave, más lento). "
                        "Sin este flag: capas sparse del probe")
    p.add_argument("--output_dir",  default="speed_probe/output/vis")
    p.add_argument("--fps",         type=int, default=3,
                   help="FPS del MP4. 3 fps con 24 capas = ~8 segundos")
    args = p.parse_args()

    # Límite: máximo 2 videos
    if len(args.videos) > 2:
        print("[WARN] Máx. 2 videos recomendado — usando solo los primeros 2.")
        args.videos = args.videos[:2]

    labels = list(args.labels or [])
    while len(labels) < len(args.videos):
        labels.append(os.path.splitext(os.path.basename(args.videos[len(labels)]))[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("V-JEPA 2 — Layer Clustering Visualization")
    print("=" * 60)
    print(f"  Videos ({len(args.videos)}):")
    for v, l in zip(args.videos, labels):
        print(f"    {l}: {v}")
    print(f"  Componente:  {args.component}")
    print(f"  Capas:       {'todas' if args.all_layers else 'sparse (probe layers)'}")
    print(f"  Device:      {device}")
    print("=" * 60)

    for v in args.videos:
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video no encontrado: {v}")

    # 1. Cargar modelo
    model = load_encoder(args.checkpoint, args.model_name, args.img_size,
                         args.num_frames, device)

    # 2. Decidir capas: todas (24) o sparse (12 del probe)
    num_blocks = len(model.blocks)
    if args.all_layers:
        vis_layers = list(range(num_blocks))         # todas: 0..23
    else:
        from speed_probe.activation_extractor import DEFAULT_LAYERS
        vis_layers = DEFAULT_LAYERS.get(
            num_blocks,
            sorted(set([0] + list(range(0, num_blocks, max(1, num_blocks//11))) + [num_blocks-1]))
        )
    print(f"  Capas a visualizar ({len(vis_layers)}): {vis_layers}")

    # 3. Registrar hooks solo para el componente elegido
    #    (ahorramos memoria vs hookear los 3 componentes)
    extractor = _SingleComponentExtractor(model, vis_layers, args.component)

    # 4. Extraer activaciones de token para cada video
    all_layer_acts = []
    for i, video_path in enumerate(args.videos):
        print(f"\n[{i+1}/{len(args.videos)}] Extrayendo: {labels[i]}")
        video_tensor = load_video(video_path, args.num_frames, args.img_size)
        x = video_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)
        layer_acts = {}
        for name, act in extractor.activations.items():
            layer_i = int(name.split("_")[1])
            layer_acts[layer_i] = act.squeeze(0).float().numpy()   # [N, D]
        extractor.activations.clear()
        del x, video_tensor
        gc.collect()
        all_layer_acts.append(layer_acts)

    extractor.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 5. PCA global
    print("\nAjustando PCA global...")
    pca = fit_global_pca(all_layer_acts)

    # 6. Proyectar
    print("Proyectando tokens...")
    projected = project_all(all_layer_acts, pca)
    del all_layer_acts
    gc.collect()

    # 7. Animación
    layers_sorted = sorted(projected[0].keys())
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"layer_clustering_{args.model_name}_{args.component}.mp4"
    )
    print(f"\nGenerando MP4 ({len(layers_sorted)} capas, {args.fps} fps)...")
    make_animation(projected, labels, args.num_frames, layers_sorted,
                   out_path, args.component, args.fps)

    print("\nListo.")


# ── Extractor ligero solo para un componente ─────────────────────────────────

class _SingleComponentExtractor:
    """
    Versión minimalista del extractor que hookea solo UN tipo de componente.
    Más eficiente en memoria que hookear los 3 a la vez.

    component: "residual" → hook en block
               "attn"     → hook en block.attn
               "mlp"      → hook en block.mlp
    """
    def __init__(self, model, layers, component, to_cpu=True):
        self.activations = {}
        self._handles = []
        self.to_cpu = to_cpu

        for i in layers:
            block = model.blocks[i]
            if component == "residual":
                target = block
            elif component == "attn":
                target = block.attn
            elif component == "mlp":
                target = block.mlp
            else:
                raise ValueError(f"Componente desconocido: {component}")

            name = f"block_{i}_{component}"
            handle = target.register_forward_hook(self._hook(name))
            self._handles.append(handle)

        print(f"[Extractor] {len(self._handles)} hooks · componente={component} · capas={layers}")

    def _hook(self, name):
        def fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            act = act.detach()
            if self.to_cpu:
                act = act.cpu()
            self.activations[name] = act
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations.clear()
        print("[Extractor] Hooks removidos.")


if __name__ == "__main__":
    main()
