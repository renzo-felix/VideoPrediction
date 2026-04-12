"""
run_vjepa2_layer_probing.py
============================
Entrena clasificadores lineales (Linear Probes) sobre activaciones internas
de V-JEPA 2 para predecir la etiqueta de velocidad del video.

Adaptado de run_layer_probing.py (VideoMAEv2) para la arquitectura de V-JEPA 2.

DIFERENCIAS VS VideoMAEv2:
--------------------------
  - Carga el encoder de V-JEPA 2 (ViT-Large por defecto) desde vjepa2/
  - Normalización ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Checkpoint: soporta clave "encoder" (release público) y "target_encoder" (preentrenamiento)
  - Nombre del modelo configurable: vit_large, vit_huge, vit_giant_xformers
  - Número de bloques y embed_dim se detectan automáticamente

PIPELINE:
  1. Cargar encoder V-JEPA 2 con checkpoint, modo eval(), no_grad()
  2. Cargar physical_diagnostics.csv con videos y labels de velocidad
  3. Forward pass → hooks capturan activaciones → pool → almacenar
  4. Train/test split 80/20
  5. Entrenar LogisticRegression por capa/componente
  6. Loguear a WandB y guardar JSON + NPZ

USO (SSv2 con ViT-Large):
  python run_vjepa2_layer_probing.py \\
      --checkpoint /home/projects/video-prediction/checkpoints/vjepa2/vitl.pt \\
      --model_name vit_large \\
      --csv physical_diagnostics.csv \\
      --data_root dataset/ssv2_luis \\
      --num_frames 16 \\
      --img_size 256 \\
      --output_dir output_dir/vjepa2_vitl_ssv2 \\
      --wandb_project vjepa2_probing

USO (K400 con ViT-Large, crear primero el CSV de K400):
  python run_vjepa2_layer_probing.py \\
      --checkpoint /home/projects/video-prediction/checkpoints/vjepa2/vitl.pt \\
      --model_name vit_large \\
      --csv physical_diagnostics_k400.csv \\
      --data_root dataset/k400 \\
      --video_format mp4 \\
      --num_frames 16 \\
      --img_size 256 \\
      --output_dir output_dir/vjepa2_vitl_k400 \\
      --wandb_project vjepa2_probing
"""

import argparse
import csv
import gc
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

# ---- Añadir ruta del repo vjepa2 al path ----
# Ajustar VJEPA2_DIR si la estructura del proyecto es diferente
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VJEPA2_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "vjepa2")
if VJEPA2_DIR not in sys.path:
    sys.path.insert(0, VJEPA2_DIR)

from vjepa2_activation_extractor import VJEPA2ActivationExtractor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[ADVERTENCIA] wandb no instalado. Se desactivará el logging a WandB.")
    WANDB_AVAILABLE = False

# Normalización ImageNet (V-JEPA 2 usa ImageNet mean/std, no [0.5, 0.5, 0.5])
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Carga de video
# ============================================================================
def load_video_frames_from_dir(video_dir: str, num_frames: int = 16, img_size: int = 256) -> torch.Tensor:
    """
    Carga frames de un directorio (formato SSv2: frames como imágenes JPG/PNG).

    Returns:
        Tensor [3, num_frames, img_size, img_size] = [C, T, H, W]
        Normalizado con ImageNet mean/std.
    """
    from torchvision import transforms

    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(frame_files) == 0:
        raise FileNotFoundError(f"No se encontraron frames en {video_dir}")

    total = len(frame_files)
    indices = (
        np.linspace(0, total - 1, num_frames, dtype=int)
        if total >= num_frames
        else np.array(list(range(total)) + [total - 1] * (num_frames - total))
    )

    transform = transforms.Compose([
        transforms.Resize(int(256.0 / 224 * img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    frames = []
    for idx in indices:
        with Image.open(os.path.join(video_dir, frame_files[idx])) as img:
            frames.append(transform(img.convert("RGB")))

    # [T, 3, H, W] → [3, T, H, W]
    return torch.stack(frames, dim=0).permute(1, 0, 2, 3)


def load_video_frames_from_mp4(video_path: str, num_frames: int = 16, img_size: int = 256) -> torch.Tensor:
    """
    Carga frames de un archivo MP4/AVI (formato K400 y otros).
    Requiere: pip install decord

    Returns:
        Tensor [3, num_frames, img_size, img_size] = [C, T, H, W]
        Normalizado con ImageNet mean/std.
    """
    try:
        from decord import VideoReader
    except ImportError:
        raise ImportError("Instalar decord: pip install decord")

    from torchvision import transforms

    vr = VideoReader(video_path)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_np = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

    transform = transforms.Compose([
        transforms.Resize(int(256.0 / 224 * img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    frames = []
    for i in range(frames_np.shape[0]):
        img = Image.fromarray(frames_np[i])
        frames.append(transform(img))

    return torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]


# ============================================================================
# Carga del modelo V-JEPA 2
# ============================================================================
def load_vjepa2_encoder(checkpoint_path: str, model_name: str, img_size: int,
                         num_frames: int, device: torch.device) -> nn.Module:
    """
    Carga el encoder de V-JEPA 2 desde un checkpoint local.

    Soporta dos formatos de checkpoint:
      - Checkpoint de release público: {"encoder": state_dict}
      - Checkpoint de preentrenamiento JEPA: {"target_encoder": state_dict}

    Args:
        model_name: "vit_large" | "vit_huge" | "vit_giant_xformers" | "vit_giant_xformers_rope"
        img_size:   Resolución cuadrada (256 para ViT-L/H, 384 para ViT-G-384)
        num_frames: Frames de entrada (16 o 64 en el repo oficial)

    Returns:
        Encoder en eval() en `device`.
    """
    from src.models.vision_transformer import (
        vit_large,
        vit_huge,
        vit_giant_xformers,
        vit_giant_xformers_rope,
    )

    MODEL_MAP = {
        "vit_large": vit_large,
        "vit_huge": vit_huge,
        "vit_giant_xformers": vit_giant_xformers,
        "vit_giant_xformers_rope": vit_giant_xformers_rope,
    }

    if model_name not in MODEL_MAP:
        raise ValueError(f"model_name debe ser uno de: {list(MODEL_MAP.keys())}")

    print(f"Inicializando modelo: {model_name} (img_size={img_size}, num_frames={num_frames})")
    model = MODEL_MAP[model_name](
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        uniform_power=True,
        use_rope=True,
    )

    print(f"Cargando checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Detectar formato del checkpoint
    if "encoder" in ckpt:
        state_dict = ckpt["encoder"]
        print("  → Formato: checkpoint de release (key='encoder')")
    elif "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
        print("  → Formato: checkpoint de preentrenamiento (key='target_encoder')")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        print("  → Formato: checkpoint genérico (key='model')")
    else:
        # Asumir que el checkpoint ES el state_dict directamente
        state_dict = ckpt
        print("  → Formato: state_dict directo")

    # Limpiar prefijos de DataParallel / backbone wrappers
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        clean_sd[k] = v

    msg = model.load_state_dict(clean_sd, strict=False)
    print(f"  → missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"  → missing (primeras 5): {msg.missing_keys[:5]}")

    model = model.to(device)
    model.eval()
    print(f"  → Modelo listo: {model.get_num_layers()} bloques, embed_dim={model.embed_dim}")
    return model


# ============================================================================
# Carga del CSV de diagnóstico
# ============================================================================
def load_diagnostic_csv(csv_path: str) -> list:
    """
    Carga physical_diagnostics.csv generado por create_physical_subset.py.

    Columnas esperadas: video_path, num_frames, original_label, class_id, speed_label
    """
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "video_path": row["video_path"],
                "num_frames": int(row["num_frames"]),
                "speed_label": int(row["speed_label"]),
                "original_label": row.get("original_label", ""),
                "class_id": int(row.get("class_id", -1)),
            })
    return entries


# ============================================================================
# Extracción de activaciones
# ============================================================================
def extract_all_activations(
    model: nn.Module,
    extractor: VJEPA2ActivationExtractor,
    entries: list,
    data_root: str,
    num_frames: int,
    img_size: int,
    device: torch.device,
    video_format: str = "frames",
    max_videos: int = None,
) -> dict:
    """
    Extrae activaciones layer-wise para todos los videos del CSV.

    Args:
        video_format: "frames" para SSv2 (directorio de imágenes) o "mp4" para K400.
        max_videos: Limitar cantidad de videos (útil para debug).

    Returns:
        Dict { nombre_activacion: {"features": np.array [N, D], "labels": np.array [N]} }
    """
    all_pooled = {}
    all_labels = []

    if max_videos:
        entries = entries[:max_videos]

    total = len(entries)
    errors = 0

    for i, entry in enumerate(entries):
        video_path = os.path.join(data_root, entry["video_path"])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {entry['video_path']}")

        try:
            # Cargar video → [C, T, H, W]
            if video_format == "frames":
                video_tensor = load_video_frames_from_dir(video_path, num_frames, img_size)
            else:
                video_tensor = load_video_frames_from_mp4(video_path, num_frames, img_size)

            # Añadir dim de batch → [1, C, T, H, W]
            video_tensor = video_tensor.unsqueeze(0).to(device)

            # Forward pass (hooks capturan automáticamente)
            with torch.no_grad():
                _ = model(video_tensor)

            # Recoger activaciones pooled [1, D] por capa
            pooled = extractor.get_pooled_activations()
            for name, tensor in pooled.items():
                if name not in all_pooled:
                    all_pooled[name] = []
                all_pooled[name].append(tensor.squeeze(0).numpy().copy())

            all_labels.append(entry["speed_label"])
            extractor.clear_activations()
            del video_tensor

            if (i + 1) % 100 == 0:
                gc.collect()

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [ERROR] {entry['video_path']}: {e}")
            elif errors == 6:
                print("  [SUPRIMIDO] Más errores suprimidos...")
            extractor.clear_activations()
            continue

    print(f"\n  Procesados: {len(all_labels)}/{total} videos ({errors} errores)")

    labels_array = np.array(all_labels)
    result = {}
    for name, feat_list in all_pooled.items():
        result[name] = {
            "features": np.stack(feat_list, axis=0),  # [N, D]
            "labels": labels_array,
        }
    return result


# ============================================================================
# Linear Probing
# ============================================================================
def train_probes(activations_dict: dict, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Entrena un LogisticRegression por cada capa/componente.

    Returns:
        Dict { nombre: {"accuracy": float, "confusion_matrix": list,
                         "train_size": int, "test_size": int} }
    """
    results = {}
    for name, data in sorted(activations_dict.items()):
        features = data["features"]  # [N, D]
        labels = data["labels"]      # [N]

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=random_state)
        clf.fit(X_train, y_train)

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
def log_to_wandb(results: dict, project: str, run_name: str = None, config: dict = None):
    """
    Loguea resultados a Weights & Biases.

    Genera:
      - Métricas probe_accuracy por capa y componente
      - Gráfica layer-wise accuracy (eje X = capa, eje Y = accuracy, color = componente)
    """
    if not WANDB_AVAILABLE:
        print("[SKIP] WandB no disponible")
        return

    wandb.init(project=project, name=run_name or "vjepa2_layer_probing", config=config)

    for name, data in sorted(results.items()):
        wandb.log({f"probe_accuracy/{name}": data["accuracy"]})

    # Tabla para gráfica layer-wise
    table_data = []
    for name, data in sorted(results.items()):
        parts = name.split("_")
        layer_idx = int(parts[1])
        component = parts[2]  # "residual", "attn", "mlp"
        table_data.append([layer_idx, component, data["accuracy"]])

    table = wandb.Table(data=table_data, columns=["layer", "component", "accuracy"])
    wandb.log({
        "layer_wise_probe_accuracy": wandb.plot.line(
            table, "layer", "accuracy",
            stroke="component",
            title="V-JEPA 2 Layer-wise Probe Accuracy (Speed Detection)"
        )
    })

    wandb.finish()
    print("Resultados logueados a WandB.")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Linear Probing Layer-wise sobre V-JEPA 2")

    # Modelo
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Ruta al checkpoint del encoder V-JEPA 2 (.pt)")
    parser.add_argument("--model_name", type=str, default="vit_large",
                        choices=["vit_large", "vit_huge", "vit_giant_xformers", "vit_giant_xformers_rope"],
                        help="Variante de ViT a cargar")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Resolución de entrada (256 para ViT-L/H, 384 para ViT-G-384)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Número de frames por video")

    # Datos
    parser.add_argument("--csv", type=str, default="physical_diagnostics.csv",
                        help="CSV de diagnóstico (video_path, speed_label)")
    parser.add_argument("--data_root", type=str, default="dataset/ssv2_luis",
                        help="Ruta raíz de los videos")
    parser.add_argument("--video_format", type=str, default="frames",
                        choices=["frames", "mp4"],
                        help="'frames' para SSv2 (dirs de imgs), 'mp4' para K400/Kinetics")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limitar videos (para debug)")

    # Capas
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Capas a analizar (default: distribución uniforme automática)")

    # Salida
    parser.add_argument("--output_dir", type=str, default="output_dir/vjepa2",
                        help="Directorio para guardar activaciones.npz y probing_results.json")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="vjepa2_probing",
                        help="Proyecto de WandB")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Nombre del run (default: auto)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Desactivar WandB")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Cargar encoder
    model = load_vjepa2_encoder(
        args.checkpoint, args.model_name, args.img_size, args.num_frames, device
    )

    # 2. Configurar extractor
    extractor = VJEPA2ActivationExtractor(
        model, layers=args.layers, detach=True, to_cpu=True
    )

    # 3. Cargar CSV
    print(f"\nCargando CSV: {args.csv}")
    entries = load_diagnostic_csv(args.csv)
    fast = sum(1 for e in entries if e["speed_label"] == 1)
    slow = sum(1 for e in entries if e["speed_label"] == 0)
    print(f"  → {len(entries)} videos ({fast} rápidos, {slow} lentos)")

    # 4. Extraer activaciones
    num_layers = len(extractor.layers)
    print(f"\nExtrayendo activaciones ({num_layers} capas × 3 componentes)...")
    activations_dict = extract_all_activations(
        model, extractor, entries,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        device=device,
        video_format=args.video_format,
        max_videos=args.max_videos,
    )

    # Limpiar GPU
    extractor.remove_hooks()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 5. Guardar activaciones comprimidas en disco
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "activations.npz")
    print(f"\nGuardando activaciones en: {save_path}")
    save_dict = {}
    for name, data in activations_dict.items():
        save_dict[f"{name}_features"] = data["features"]
    save_dict["labels"] = list(activations_dict.values())[0]["labels"]
    np.savez_compressed(save_path, **save_dict)
    print("  → Guardado.")

    # 6. Entrenar Linear Probes
    print(f"\nEntrenando Linear Probes ({len(activations_dict)} combinaciones)...")
    results = train_probes(activations_dict)

    # 7. Guardar resultados JSON
    results_path = os.path.join(args.output_dir, "probing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados: {results_path}")

    # 8. Loguear a WandB
    if not args.no_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"vjepa2_{args.model_name}_{os.path.basename(args.output_dir)}"
        config = {
            "model_name": args.model_name,
            "checkpoint": args.checkpoint,
            "img_size": args.img_size,
            "num_frames": args.num_frames,
            "video_format": args.video_format,
            "num_videos": len(entries),
            "layers": extractor.layers,
        }
        log_to_wandb(results, args.wandb_project, run_name, config)

    # 9. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN POR COMPONENTE")
    print("=" * 60)
    for component in ["residual", "attn", "mlp"]:
        print(f"\n  {component.upper()}:")
        for name, data in sorted(results.items()):
            if name.endswith(f"_{component}"):
                layer = name.split("_")[1]
                print(f"    Capa {layer:>2s}: {data['accuracy']:.4f}")


if __name__ == "__main__":
    main()
