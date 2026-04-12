"""
run_speed_probe.py
==================
Linear probe de velocidad (rápido/lento) sobre activaciones layer-wise de V-JEPA 2.
Adaptado del pipeline de Luis (VideoMAEv2_Proyecto_Paper/run_layer_probing.py).

Uso (SSv2 .webm):
  python speed_probe/run_speed_probe.py \
      --checkpoint checkpoints/vitl.pt \
      --csv /home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/physical_diagnostics.csv \
      --data_root data/ssv2/videos/20bn-something-something-v2 \
      --video_format webm \
      --output_dir speed_probe/output/vitl_ssv2

Uso (K400 .mp4):
  python speed_probe/run_speed_probe.py \
      --checkpoint checkpoints/vitl.pt \
      --csv speed_probe/physical_diagnostics_k400.csv \
      --data_root /home/datasets/k400/train \
      --video_format mp4 \
      --output_dir speed_probe/output/vitl_k400
"""

import argparse, csv, gc, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Importar desde el repo vjepa2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed_probe.activation_extractor import VJEPA2ActivationExtractor
from src.models.vision_transformer import vit_large, vit_huge, vit_giant_xformers_rope

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_MAP = {
    "vit_large": vit_large,
    "vit_huge":  vit_huge,
    "vit_giant": vit_giant_xformers_rope,
}

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False


# ── Carga de video ────────────────────────────────────────────────────────────

def load_video(path: str, num_frames: int, img_size: int) -> torch.Tensor:
    """Carga .webm / .mp4 → [C, T, H, W] normalizado ImageNet.
    Intenta decord (num_threads=1) primero; fallback a cv2 si falla.
    """
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(256.0/224*img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Intento 1: decord con un solo thread (evita EAGAIN en .webm) ──────────
    try:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(path, ctx=decord_cpu(0), num_threads=1)
        idx = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        frames_np = vr.get_batch(idx).asnumpy()          # [T, H, W, C]
        frames = [tf(Image.fromarray(frames_np[i])) for i in range(frames_np.shape[0])]
        return torch.stack(frames).permute(1, 0, 2, 3)   # [C, T, H, W]
    except Exception:
        pass

    # ── Fallback: cv2.VideoCapture ────────────────────────────────────────────
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Algunos .webm no reportan frame count; leemos todos y submuestreamos
        raw = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(raw) == 0:
            raise IOError(f"Video vacío: {path}")
        indices = np.linspace(0, len(raw) - 1, num_frames, dtype=int)
        pil_frames = [Image.fromarray(raw[i]) for i in indices]
    else:
        indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        pil_frames, fi = [], 0
        while cap.isOpened() and len(pil_frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices:
                pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            fi += 1
        cap.release()

    if len(pil_frames) == 0:
        raise IOError(f"No se extrajeron frames de: {path}")
    # Si faltan frames al final, repetimos el último
    while len(pil_frames) < num_frames:
        pil_frames.append(pil_frames[-1])

    frames = [tf(f) for f in pil_frames[:num_frames]]
    return torch.stack(frames).permute(1, 0, 2, 3)   # [C, T, H, W]


def resolve_path(data_root: str, csv_path: str, ext: str) -> str:
    """
    Convierte path del CSV (ej. "SomethingV2/frames/34899") a ruta real.
    Extrae el ID del último componente y añade la extensión.
    """
    video_id = os.path.basename(csv_path)
    return os.path.join(data_root, f"{video_id}.{ext}")


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_encoder(checkpoint: str, model_name: str, img_size: int,
                 num_frames: int, device) -> nn.Module:
    print(f"Cargando {model_name} desde {checkpoint}...")
    model = MODEL_MAP[model_name](
        img_size=(img_size, img_size), num_frames=num_frames,
        patch_size=16, tubelet_size=2, use_sdpa=True,
        uniform_power=True, use_rope=True,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Soporta key "encoder" (release) y "target_encoder" (preentrenamiento)
    key = "encoder" if "encoder" in ckpt else "target_encoder" if "target_encoder" in ckpt else None
    sd = ckpt[key] if key else ckpt
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model.to(device).eval()


# ── Extracción de activaciones ────────────────────────────────────────────────

def _load_one(args_tuple):
    """Worker para ThreadPoolExecutor: devuelve (tensor, label) o (None, label) si falla."""
    path, label, num_frames, img_size = args_tuple
    try:
        t = load_video(path, num_frames, img_size)
        return (t, label, None)
    except Exception as e:
        return (None, label, str(e))


def extract_activations(model, extractor, entries, data_root, num_frames,
                        img_size, device, ext, max_videos=None,
                        batch_size=8, num_workers=4):
    """
    Extrae activaciones con:
      - Carga paralela de videos (ThreadPoolExecutor, num_workers)
      - Inferencia en batches (batch_size videos por forward pass)
    En un A100 con batch_size=8 esto es ~6-8x más rápido que batch_size=1.
    """
    all_pooled, all_labels = {}, []
    if max_videos:
        entries = entries[:max_videos]

    n = len(entries)
    errors = 0
    processed = 0

    # Iterar en chunks de batch_size, cargando con threads
    for batch_start in range(0, n, batch_size):
        batch_entries = entries[batch_start: batch_start + batch_size]

        # Preparar args para carga paralela
        load_args = [
            (resolve_path(data_root, e["video_path"], ext),
             e["speed_label"], num_frames, img_size)
            for e in batch_entries
        ]

        # Cargar en paralelo
        results = [None] * len(load_args)
        with ThreadPoolExecutor(max_workers=min(num_workers, len(load_args))) as pool:
            future_to_idx = {pool.submit(_load_one, a): i
                             for i, a in enumerate(load_args)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        # Filtrar los que cargaron bien y hacer batch
        tensors, labels_batch = [], []
        for tensor, label, err in results:
            if tensor is not None:
                tensors.append(tensor)
                labels_batch.append(label)
            else:
                errors += 1

        if not tensors:
            continue

        # Inferencia en batch
        x = torch.stack(tensors).to(device)   # [B, C, T, H, W]
        with torch.no_grad():
            model(x)

        pooled = extractor.get_pooled_activations()   # {name: [B, D]}
        for name, feat in pooled.items():
            # feat: Tensor [B, D] en CPU (to_cpu=True en el extractor)
            all_pooled.setdefault(name, []).extend(
                [feat[j].numpy().copy() for j in range(feat.shape[0])]
            )
        all_labels.extend(labels_batch)
        extractor.clear_activations()
        del x, tensors
        processed += len(labels_batch)

        if processed % 500 == 0 or batch_start == 0:
            pct = 100 * processed / n
            print(f"  [{processed}/{n}]  {pct:.1f}%  errores={errors}")
            gc.collect()

    print(f"  Extracción completa: {processed} ok, {errors} errores de {n} total")
    labels_arr = np.array(all_labels)
    return {name: {"features": np.stack(fs), "labels": labels_arr}
            for name, fs in all_pooled.items()}


# ── Linear Probing ────────────────────────────────────────────────────────────

def train_probes(data: dict, test_size=0.2, seed=42):
    results = {}
    for name, d in sorted(data.items()):
        X_tr, X_te, y_tr, y_te = train_test_split(
            d["features"], d["labels"], test_size=test_size,
            random_state=seed, stratify=d["labels"])
        clf = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0, random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        results[name] = {
            "accuracy": float(acc),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
            "train_size": len(X_tr), "test_size": len(X_te),
        }
        print(f"  {name}: {acc:.4f}")
    return results


# ── WandB ─────────────────────────────────────────────────────────────────────

def log_wandb(results, project, run_name, config):
    if not WANDB_OK: return
    wandb.init(project=project, name=run_name, config=config)
    table = []
    for name, d in sorted(results.items()):
        parts = name.split("_")
        table.append([int(parts[1]), parts[2], d["accuracy"]])
        wandb.log({f"probe_accuracy/{name}": d["accuracy"]})
    t = wandb.Table(data=table, columns=["layer", "component", "accuracy"])
    wandb.log({"layer_wise_accuracy": wandb.plot.line(
        t, "layer", "accuracy", stroke="component",
        title="V-JEPA 2 Speed Probe — Layer-wise Accuracy")})
    wandb.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,
                   help="Ruta al encoder vitl.pt (NO es el probe de clasificacion)")
    p.add_argument("--model_name",   required=True,
                   choices=["vit_large", "vit_huge", "vit_giant"])
    p.add_argument("--img_size",     type=int, required=True,
                   help="256 para ViT-L/H, 384 para ViT-G-384")
    p.add_argument("--num_frames",   type=int, required=True,
                   help="Frames a samplear por video (16 recomendado)")
    p.add_argument("--csv",          required=True,
                   help="physical_diagnostics.csv con speed_label")
    p.add_argument("--data_root",    required=True,
                   help="Carpeta con los .webm o .mp4")
    p.add_argument("--video_format", required=True,
                   choices=["webm", "mp4"],
                   help="webm para SSv2, mp4 para K400")
    p.add_argument("--layers",       type=int, nargs="+", default=None)
    p.add_argument("--max_videos",   type=int, default=None)
    p.add_argument("--batch_size",   type=int, default=8,
                   help="Videos por forward pass en GPU. A100 40GB: 8-16 (default 8)")
    p.add_argument("--num_workers",  type=int, default=4,
                   help="Threads para carga paralela de video (default 4)")
    p.add_argument("--output_dir",   default="speed_probe/output")
    p.add_argument("--wandb_project", default="vjepa2_speed_probe")
    p.add_argument("--wandb_run",   default=None)
    p.add_argument("--no_wandb",    action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("V-JEPA 2 — Speed Linear Probe")
    print("=" * 60)
    print(f"  Encoder (frozen):  {args.checkpoint}")
    print(f"  Modelo:            {args.model_name}")
    print(f"  Resolución:        {args.img_size}px  |  Frames: {args.num_frames}")
    print(f"  CSV speed labels:  {args.csv}")
    print(f"  Videos root:       {args.data_root}  [{args.video_format}]")
    print(f"  Output:            {args.output_dir}")
    print(f"  Device:            {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    print("=" * 60)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV no encontrado: {args.csv}")
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"data_root no encontrado: {args.data_root}")

    # 1. Modelo
    model = load_encoder(args.checkpoint, args.model_name, args.img_size, args.num_frames, device)

    # 2. Extractor
    extractor = VJEPA2ActivationExtractor(model, layers=args.layers)

    # 3. CSV
    entries = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            entries.append({"video_path": row["video_path"],
                            "speed_label": int(row["speed_label"])})
    fast = sum(1 for e in entries if e["speed_label"]==1)
    print(f"CSV: {len(entries)} videos ({fast} rápidos, {len(entries)-fast} lentos)")

    # 4. Extraer activaciones
    print(f"\nExtrayendo activaciones...")
    print(f"  Batch size GPU:    {args.batch_size}  |  Workers carga: {args.num_workers}")
    act = extract_activations(model, extractor, entries, args.data_root,
                              args.num_frames, args.img_size, device,
                              args.video_format, args.max_videos,
                              args.batch_size, args.num_workers)
    extractor.remove_hooks()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    gc.collect()

    # 5. Guardar .npz
    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "activations.npz")
    save = {f"{k}_features": v["features"] for k, v in act.items()}
    save["labels"] = list(act.values())[0]["labels"]
    np.savez_compressed(npz_path, **save)
    print(f"Activaciones guardadas: {npz_path}")

    # 6. Probes
    print("\nEntrenando linear probes...")
    results = train_probes(act)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados: {json_path}")

    # 7. WandB
    if not args.no_wandb:
        run_name = args.wandb_run or f"vjepa2_{args.model_name}_{args.video_format}"
        log_wandb(results, args.wandb_project, run_name,
                  {"model": args.model_name, "format": args.video_format,
                   "n_videos": len(entries)})

    # 8. Resumen
    print("\n" + "="*50)
    for comp in ["residual", "attn", "mlp"]:
        print(f"\n{comp.upper()}:")
        for name, d in sorted(results.items()):
            if name.endswith(f"_{comp}"):
                print(f"  capa {name.split('_')[1]:>2}: {d['accuracy']:.4f}")


if __name__ == "__main__":
    main()
