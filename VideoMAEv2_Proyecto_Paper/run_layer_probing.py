"""
run_layer_probing.py
====================
Entrena clasificadores lineales (Linear Probes) sobre las activaciones internas
de VideoMAEv2 (ViT-Giant) para predecir la etiqueta de velocidad.

JUSTIFICACIÓN ARQUITECTÓNICA:
-----------------------------
Un "Linear Probe" es un clasificador lineal (LogisticRegression) entrenado
sobre las activaciones congeladas de una capa intermedia. Si el probe obtiene
alta accuracy en una capa, significa que la información de velocidad es
linealmente decodificable en esa representación — es decir, el modelo YA
codifica esa propiedad física en esa capa.

La gráfica "Layer-wise Probe Accuracy" revela:
  - Si la velocidad se codifica temprano (capas 0-8: features de bajo nivel)
  - Si emerge en capas medias (12-24: abstracciones espaciotemporales)
  - Si aparece tardíamente (28-39: semántica de acción)

PIPELINE:
  1. Cargar modelo ViT-Giant con checkpoint SSv2, modo eval(), no_grad()
  2. Cargar physical_diagnostics.csv con videos y labels de velocidad
  3. Forward pass de cada video → hooks capturan activaciones → pool → almacenar
  4. Train/test split 80/20 de las activaciones pooled
  5. Entrenar LogisticRegression por capa/componente
  6. Loguear results a WandB y guardar JSON

USO:
  python run_layer_probing.py \
      --checkpoint /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
      --csv physical_diagnostics.csv \
      --batch_size 4 \
      --num_frames 16 \
      --wandb_project videomae_probing
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Importar modelo y hooks
from models import vit_giant_patch14_224
from mechanistic_hooks import VideoMAEActivationExtractor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[ADVERTENCIA] wandb no instalado. Se desactivará el logging a WandB.")
    WANDB_AVAILABLE = False


# ============================================================================
# Carga de video como tensor
# ============================================================================
def load_video_frames(video_dir: str, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    """
    Carga frames de un directorio de video (formato SSv2: frames como imágenes).

    SSv2 almacena cada video como una carpeta con frames individuales como imágenes.
    Los frames se nombran como 00001.jpg, 00002.jpg, etc.

    Args:
        video_dir: Ruta al directorio del video con frames
        num_frames: Número de frames a muestrear uniformemente
        size: Tamaño de resize (224 para ViT-Giant)

    Returns:
        Tensor de forma [3, num_frames, size, size] (C, T, H, W)
        Normalizado con mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    """
    from torchvision import transforms

    # Encontrar todos los frames (jpg/png)
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(frame_files) == 0:
        raise FileNotFoundError(f"No se encontraron frames en {video_dir}")

    # Muestreo uniforme de num_frames frames
    total_frames = len(frame_files)
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Si hay menos frames que los requeridos, repetir el último
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(total_frames - 1)
        indices = np.array(indices)

    # Transform: resize + center crop + normalize
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for idx in indices:
        frame_path = os.path.join(video_dir, frame_files[idx])
        with Image.open(frame_path) as img:
            rgb_img = img.convert("RGB")
            # transform: [3, 224, 224]
            frames.append(transform(rgb_img))

    # frames: lista de [3, 224, 224] → stack → [num_frames, 3, 224, 224]
    video_tensor = torch.stack(frames, dim=0)
    # Reordenar a [3, num_frames, 224, 224] = [C, T, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Carga el ViT-Giant con pesos del checkpoint SSv2 finetuned.

    Returns:
        Modelo en modo eval() en el device especificado.
    """
    print(f"Cargando modelo vit_giant_patch14_224...")
    model = vit_giant_patch14_224(
        num_classes=174,   # SSv2 tiene 174 clases
        all_frames=16,     # 16 frames de entrada
        tubelet_size=2,    # tubelet estándar
        cos_attn=True,     # ViT-Giant usa CosAttention
    )

    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # El checkpoint puede tener los pesos bajo "model" o directamente
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint

    # Limpiar prefijos "module." si vienen de DataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("module.", "")
        clean_state_dict[clean_key] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)
    print(f"  → Carga de pesos: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    model = model.to(device)
    model.eval()
    print(f"  → Modelo en eval() en {device}")
    print(f"  → {model.get_num_layers()} bloques, embed_dim={model.embed_dim}")
    return model


def load_diagnostic_csv(csv_path: str) -> list:
    """Carga physical_diagnostics.csv generado por create_physical_subset.py."""
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "video_path": row["video_path"],
                "num_frames": int(row["num_frames"]),
                "speed_label": int(row["speed_label"]),
                "original_label": row["original_label"],
                "class_id": int(row["class_id"]),
            })
    return entries


# ============================================================================
# Extracción de activaciones
# ============================================================================
def extract_all_activations(
    model: nn.Module,
    extractor: VideoMAEActivationExtractor,
    entries: list,
    data_root: str,
    num_frames: int,
    batch_size: int,
    device: torch.device,
    max_videos: int = None,
) -> dict:
    """
    Extrae activaciones para todos los videos del CSV diagnóstico.

    ESTRATEGIA DE MEMORIA (CRÍTICA PARA 40 CAPAS):
    En vez de acumular listas de arrays numpy (que fragmentan la RAM de Python
    y causaron OOM en Job 29937 con 64GB), pre-asignamos arrays numpy contiguos
    de tamaño [N_videos, embed_dim]. Esto reduce el consumo de:
      - Listas: ~40-50 GB (21K × 120 listas × overhead de Python objects)
      - Pre-asignado: ~14.3 GB (21K × 120 × 1408 × 4 bytes, contiguos en RAM)

    Para cada video:
      1. Cargar frames → tensor [C, T, H, W]
      2. Forward pass (con no_grad) → hooks capturan activaciones
      3. Pool sobre tokens → [1, 1408] por capa/componente
      4. Escribir directamente en el array pre-asignado (sin append)

    Args:
        max_videos: Si no es None, limitar la cantidad de videos (para debug)

    Returns:
        Dict con keys = nombre de activación (ej. "block_0_residual")
        y values = dict con "features" (np.array [N, 1408]) y "labels" (np.array [N])
    """
    import gc

    if max_videos:
        entries = entries[:max_videos]

    total = len(entries)
    errors = 0

    # --- PRE-ASIGNACIÓN DE ARRAYS NUMPY ---
    # Hacemos un forward pass de prueba con el primer video para descubrir
    # los nombres de las activaciones y la dimensión embed_dim.
    # Esto evita hardcodear nombres y es compatible con cualquier cantidad de capas.
    first_video_path = os.path.join(data_root, entries[0]["video_path"])
    probe_tensor = load_video_frames(first_video_path, num_frames=num_frames)
    probe_tensor = probe_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(probe_tensor)
    probe_pooled = extractor.get_pooled_activations()

    # Obtener nombres y embed_dim del forward pass de prueba
    activation_names = sorted(probe_pooled.keys())
    embed_dim = probe_pooled[activation_names[0]].squeeze(0).shape[0]  # 1408
    print(f"  Pre-asignando {len(activation_names)} arrays de [{total}, {embed_dim}] "
          f"({len(activation_names) * total * embed_dim * 4 / 1e9:.1f} GB)...")

    # Pre-asignar arrays contiguos: mucho más eficiente que listas de append
    all_features = {name: np.empty((total, embed_dim), dtype=np.float32) for name in activation_names}
    all_labels = np.empty(total, dtype=np.int64)

    # Guardar el primer video (ya lo procesamos)
    for name, tensor in probe_pooled.items():
        all_features[name][0] = tensor.squeeze(0).numpy()
    all_labels[0] = entries[0]["speed_label"]
    extractor.clear_activations()
    del probe_tensor, probe_pooled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Índice de escritura (empezamos en 1 porque el video 0 ya se procesó)
    write_idx = 1

    for i, entry in enumerate(entries[1:], start=1):
        video_path = os.path.join(data_root, entry["video_path"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Procesando video {i+1}/{total}: {entry['video_path']}")

        # Monitoreo de memoria cada 500 videos para detectar leaks temprano
        if (i + 1) % 500 == 0:
            import psutil
            mem = psutil.Process().memory_info()
            print(f"  [MEM] Video {i+1}: RSS={mem.rss / 1e9:.1f} GB, "
                  f"VMS={mem.vms / 1e9:.1f} GB")

        try:
            # 1. Cargar video → [C, T, H, W] = [3, 16, 224, 224]
            video_tensor = load_video_frames(video_path, num_frames=num_frames)
            # Añadir dimensión de batch → [1, C, T, H, W] = [1, 3, 16, 224, 224]
            video_tensor = video_tensor.unsqueeze(0).to(device)

            # 2. Forward pass (los hooks capturan automáticamente)
            with torch.no_grad():
                _ = model(video_tensor)

            # 3. Obtener activaciones pooled: [1, 1408] por capa/componente
            pooled = extractor.get_pooled_activations()

            # 4. Escribir directamente en los arrays pre-asignados (sin append)
            # Esto evita la fragmentación de RAM que causó OOM en Job 29937
            for name, tensor in pooled.items():
                all_features[name][write_idx] = tensor.squeeze(0).numpy()

            all_labels[write_idx] = entry["speed_label"]
            write_idx += 1

            # 5. Limpiar activaciones para el siguiente video
            extractor.clear_activations()
            del video_tensor, pooled

            if (i + 1) % 50 == 0:
                # Forzamos gc.collect() + empty_cache() cada 50 videos porque
                # 40 capas × 3 componentes generan ~120 tensores de [1, 2048, 1408]
                # por video. Sin limpieza periódica, la fragmentación de VRAM
                # en la RTX A6000 (48GB) provoca errores OOM en videos posteriores.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [ERROR] Video {entry['video_path']}: {e}")
            elif errors == 6:
                print(f"  [SUPRIMIDO] Se suprimirán errores adicionales...")
            extractor.clear_activations()
            continue

    print(f"\n  Procesados: {write_idx}/{total} videos ({errors} errores)")

    # Recortar arrays si hubo errores (write_idx < total)
    # Esto evita enviar filas de ceros al clasificador lineal
    result = {}
    labels_array = all_labels[:write_idx]
    for name in activation_names:
        result[name] = {
            "features": all_features[name][:write_idx],
            "labels": labels_array,
        }
    return result


# ============================================================================
# Linear Probing
# ============================================================================
def train_probes(
    activations_dict: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Entrena un LogisticRegression por cada capa/componente.

    Args:
        activations_dict: Output de extract_all_activations()
        test_size: Fracción para test split
        random_state: Seed para reproducibilidad

    Returns:
        Dict con resultados por capa/componente:
        {name: {"accuracy": float, "confusion_matrix": list, "train_size": int, "test_size": int}}
    """
    results = {}

    for name, data in sorted(activations_dict.items()):
        features = data["features"]  # [N, 1408]
        labels = data["labels"]      # [N]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Entrenar LogisticRegression
        # max_iter alto porque 1408 features puede necesitar más iteraciones
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            C=1.0,
        )
        clf.fit(X_train, y_train)

        # Evaluar
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": float(acc),
            "confusion_matrix": cm,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"  {name}: accuracy={acc:.4f} (train={len(X_train)}, test={len(X_test)})")

    return results


# ============================================================================
# WandB Logging
# ============================================================================
def log_to_wandb(results: dict, project: str, run_name: str = None):
    """
    Loguea resultados a Weights & Biases.

    Genera:
      - Métrica layer_probe_accuracy por capa y componente
      - Gráfica "Layer-wise Probe Accuracy" (eje X = capa, eje Y = accuracy)
      - Matrices de confusión por capa
    """
    if not WANDB_AVAILABLE:
        print("[SKIP] WandB no disponible")
        return

    wandb.init(project=project, name=run_name or "layer_probing")

    # Loguear accuracy por capa/componente
    for name, data in sorted(results.items()):
        wandb.log({
            f"probe_accuracy/{name}": data["accuracy"],
        })

    # Crear tabla para gráfica Layer-wise Probe Accuracy
    # Parsear nombres: "block_0_residual" → layer=0, component="residual"
    table_data = []
    for name, data in sorted(results.items()):
        parts = name.split("_")
        # block_0_residual → ["block", "0", "residual"]
        layer_idx = int(parts[1])
        component = parts[2]  # "residual", "attn", o "mlp"
        table_data.append([layer_idx, component, data["accuracy"]])

    table = wandb.Table(
        data=table_data,
        columns=["layer", "component", "accuracy"]
    )

    # Gráfica line plot: accuracy vs layer, coloreado por componente
    wandb.log({
        "layer_wise_probe_accuracy": wandb.plot.line(
            table, "layer", "accuracy",
            stroke="component",
            title="Layer-wise Probe Accuracy (Speed Detection)"
        )
    })

    # Loguear matrices de confusión
    for name, data in results.items():
        cm = np.array(data["confusion_matrix"])
        wandb.log({
            f"confusion_matrix/{name}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=None,
                preds=None,
                class_names=["slow", "fast"],
            )
        })

    wandb.finish()
    print("✅ Resultados logueados a WandB")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Linear Probing sobre VideoMAEv2")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
                        help="Ruta al checkpoint SSv2 finetuned")
    parser.add_argument("--csv", type=str, default="physical_diagnostics.csv",
                        help="CSV de physical_diagnostics generado por create_physical_subset.py")
    parser.add_argument("--data_root", type=str, default="dataset/ssv2_luis",
                        help="Ruta raíz de los videos SSv2")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="No usado directamente (procesamos de 1 en 1 por hooks)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Número de frames por video")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limitar cantidad de videos (para debug)")
    parser.add_argument("--output_dir", type=str, default="output_dir",
                        help="Directorio de salida para probing_results.json")
    parser.add_argument("--wandb_project", type=str, default="videomae_probing",
                        help="Nombre del proyecto WandB")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Desactivar WandB logging")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=list(range(40)),
                        help="Capas a analizar (default: todas las 40 capas del ViT-Giant)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Cargar modelo
    model = load_model(args.checkpoint, device)

    # 2. Configurar extractor de activaciones
    extractor = VideoMAEActivationExtractor(
        model, layers=args.layers,
        detach=True, to_cpu=True
    )

    # 3. Cargar CSV diagnóstico
    print(f"\nCargando CSV diagnóstico: {args.csv}")
    entries = load_diagnostic_csv(args.csv)
    fast_count = sum(1 for e in entries if e["speed_label"] == 1)
    slow_count = sum(1 for e in entries if e["speed_label"] == 0)
    print(f"  → {len(entries)} videos ({fast_count} rápidos, {slow_count} lentos)")

    # 4. Extraer activaciones
    print(f"\nExtrayendo activaciones de {len(args.layers)} capas × 3 componentes...")
    activations_dict = extract_all_activations(
        model, extractor, entries,
        data_root=args.data_root,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        device=device,
        max_videos=args.max_videos,
    )

    # Limpiar hooks y liberar modelo de GPU
    extractor.remove_hooks()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5. Guardar activaciones en disco (para el visualizador de clustering)
    os.makedirs(args.output_dir, exist_ok=True)
    # Naming dinámico basado en la cantidad de capas para no sobrescribir
    # los resultados existentes de 11 capas (activations.npz, probing_results.json).
    # Con 40 capas generamos activations_40layers.npz; con 11, activations_11layers.npz.
    num_layers = len(args.layers)
    npz_name = f"activations_{num_layers}layers.npz"
    save_path = os.path.join(args.output_dir, npz_name)
    print(f"\n[INFO] Guardando matriz de activaciones en disco ({save_path})...")
    
    save_dict = {}
    for name, data in activations_dict.items():
        save_dict[f"{name}_features"] = data["features"]
    # Tomamos las labels de cualquiera (son idénticas)
    save_dict["labels"] = list(activations_dict.values())[0]["labels"]
    
    # Usamos np.savez_compressed para compresión zlib (más ligereza en disco)
    np.savez_compressed(save_path, **save_dict)
    print(f"✅ Activaciones extraídas y guardadas con éxito en {npz_name}.")

    # 6. Entrenar Linear Probes
    print(f"\nEntrenando Linear Probes ({len(activations_dict)} combinaciones)...")
    results = train_probes(activations_dict)

    # 6. Guardar resultados de probing
    os.makedirs(args.output_dir, exist_ok=True)
    # Naming dinámico para JSON igual que el .npz, evitando sobrescribir
    # los probing_results.json existentes de la ejecución con 11 capas.
    json_name = f"probing_results_{num_layers}layers.json"
    output_path = os.path.join(args.output_dir, json_name)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Resultados guardados en: {output_path}")

    # 7. Loguear a WandB
    if not args.no_wandb and WANDB_AVAILABLE:
        log_to_wandb(results, args.wandb_project)

    # 8. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE PROBING POR COMPONENTE")
    print("=" * 60)
    for component in ["residual", "attn", "mlp"]:
        print(f"\n  {component.upper()}:")
        for name, data in sorted(results.items()):
            if name.endswith(f"_{component}"):
                layer = name.split("_")[1]
                print(f"    Capa {layer:>2s}: {data['accuracy']:.4f}")


if __name__ == "__main__":
    main()
