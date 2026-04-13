"""
make_k400_csv.py
================
Genera physical_diagnostics_k400.csv escaneando el directorio de K400.
Etiqueta cada video como rápido (1) o lento (0) según el nombre de la clase.

Uso:
  python speed_probe/make_k400_csv.py \
      --k400_root /home/datasets/k400 \
      --split train \
      --output speed_probe/physical_diagnostics_k400.csv
"""

import argparse, csv, os

# ── Clases RÁPIDAS de K400 (speed_label = 1) ─────────────────────────────────
# Criterio: la acción típica implica movimiento corporal/objeto veloz
FAST_CLASSES = {
    "abseiling", "air drumming", "archery", "backflip",
    "basketball dunk", "biking through snow", "bouldering", "bowling",
    "boxing", "breakdancing", "bull fighting", "bungee jumping",
    "catching or throwing frisbee", "catching or throwing softball",
    "cliff diving", "cycling", "diving cliff",
    "dodgeball", "doing aerobics", "doing karate", "doing kickboxing",
    "double dutch", "dribbling basketball", "dunking basketball",
    "fencing", "figure skating", "flipping pancake",
    "gymnastics tumbling", "hammer throw", "handstand",
    "headbanging", "high jump", "hitting baseball", "hula hooping",
    "hurling (sport)", "javelin throw", "jet skiing", "jogging",
    "juggling balls", "juggling fire",
    "jumping into pool", "jumping jacks", "jumping on trampoline",
    "karate", "kicking field goal", "kicking soccer ball", "kitesurfing",
    "luge", "motocross", "mountain biking",
    "paragliding", "parkour",
    "playing badminton", "playing basketball", "playing cricket",
    "playing football", "playing ice hockey", "playing kickball",
    "playing polo", "playing squash", "playing tennis", "playing volleyball",
    "pole vault", "punching bag", "punching person (boxing)",
    "racing car", "riding a bike", "riding horse",
    "riding mechanical bull", "river kayaking",
    "rock climbing", "roller skating", "running on treadmill",
    "sailing", "shooting goal (soccer)",
    "skateboarding", "ski jumping", "skiing slalom",
    "skipping rope", "skydiving", "slam dunk",
    "snowboarding", "soccer juggling", "speed skating",
    "spinning poi", "sprinting", "surfing water",
    "swimming breast stroke", "swimming butterfly stroke",
    "table tennis shot", "throwing axe", "throwing ball",
    "throwing discus", "throwing javelin",
    "trampoline jumping", "triple jump",
    "wakeboarding", "water skiing", "weightlifting",
    "windsurfing", "wrestling",
    "long jump", "shot put", "sword fighting",
    "playing squash or racquetball",
}

# Keywords para clasificar clases desconocidas
FAST_KEYWORDS = [
    "running", "sprint", "jump", "kick", "punch", "throw", "catching",
    "boxing", "karate", "racing", "skating", "skiing", "surfing",
    "cycling", "biking", "diving", "gymnastics", "dancing", "dribbling",
    "dunking", "flipping", "swinging", "batting", "bowling", "shooting goal",
    "weightlifting", "parkour", "bungee", "skydiving", "snowboard",
    "somersault", "backflip", "headbang", "drumming fast",
]

SLOW_KEYWORDS = [
    "sitting", "reading", "writing", "knitting", "sewing", "cooking",
    "eating", "drinking", "sleeping", "yoga", "drawing", "painting",
    "assembling", "typing", "chess", "massaging", "grooming",
    "brushing", "gardening", "fishing", "meditating", "standing",
    "talking", "singing", "playing piano", "playing guitar", "playing violin",
    "playing accordion", "playing cello", "playing harp",
    "folding", "ironing", "washing", "cleaning", "mopping",
    "arranging", "decorating", "knitting", "crocheting",
    "counting", "reading", "writing", "texting", "using computer",
]


def classify(class_name: str) -> int:
    """Devuelve 1 (rápido) o 0 (lento) para un nombre de clase K400."""
    name = class_name.lower().replace("_", " ")
    if name in FAST_CLASSES:
        return 1
    for kw in FAST_KEYWORDS:
        if kw in name:
            return 1
    for kw in SLOW_KEYWORDS:
        if kw in name:
            return 0
    # Por defecto: lento (conservador)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k400_root", required=True,
                   help="Directorio raíz de K400 (ej. /home/datasets/k400)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--output", default="speed_probe/physical_diagnostics_k400.csv")
    args = p.parse_args()

    split_dir = os.path.join(args.k400_root, args.split)
    if not os.path.exists(split_dir):
        # Algunos clusters tienen los videos directo en k400/ sin split
        split_dir = args.k400_root
        print(f"[WARN] No se encontró split '{args.split}', usando {split_dir}")

    rows = []
    class_stats = {}

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = classify(class_name)
        class_stats[class_name] = {"label": label, "count": 0}

        for fname in os.listdir(class_dir):
            if not fname.endswith(".mp4"):
                continue
            video_id = fname[:-4]   # sin .mp4
            video_path = f"{class_name}/{video_id}"
            rows.append({"video_path": video_path, "speed_label": label,
                         "class_name": class_name})
            class_stats[class_name]["count"] += 1

    if not rows:
        print(f"[ERROR] No se encontraron .mp4 en {split_dir}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "speed_label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    fast = sum(1 for r in rows if r["speed_label"] == 1)
    slow = len(rows) - fast
    print(f"CSV generado: {args.output}")
    print(f"  Total: {len(rows)} videos | Rápidos: {fast} | Lentos: {slow}")
    print(f"  Clases: {len(class_stats)}")
    print(f"\nClases RÁPIDAS:")
    for c, s in sorted(class_stats.items()):
        if s["label"] == 1:
            print(f"  {c}: {s['count']} videos")
    print(f"\nClases LENTAS (muestra):")
    shown = 0
    for c, s in sorted(class_stats.items()):
        if s["label"] == 0 and shown < 10:
            print(f"  {c}: {s['count']} videos")
            shown += 1


if __name__ == "__main__":
    main()
