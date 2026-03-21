"""
create_physical_subset.py
=========================
Genera un CSV con un subconjunto diagnóstico del dataset Something-Something V2
para estudiar la propiedad física de VELOCIDAD en VideoMAEv2 (ViT-Giant).

JUSTIFICACIÓN ARQUITECTÓNICA:
-----------------------------
SSv2 no tiene etiquetas explícitas de velocidad ("fast"/"slow"). En su lugar,
usamos pares de categorías como PROXY de velocidad:

  - Las acciones "rápidas" implican movimiento brusco, transferencia de energía
    cinética alta, o desplazamiento acelerado (ej. "Throwing", "falling like a rock").
  - Las acciones "lentas" implican movimiento suave, bajo desplazamiento, o
    ausencia de aceleración (ej. "Holding", "Poking lightly", "falling like a feather").

Este enfoque está inspirado en la técnica de "contrastive labeling" usada en
interpretabilidad mecanicista (ver: Linear Probes en ARENA exercises), donde
se necesita una señal binaria clara para entrenar un clasificador lineal sobre
las activaciones internas del modelo.

PARES DE CATEGORÍAS SELECCIONADOS:
----------------------------------
ALTA VELOCIDAD (speed_label=1):
  - 135: "Something falling like a rock"           → caída con aceleración gravitatoria máxima
  - 151: "Throwing something"                       → transferencia de energía cinética directa
  - 152: "Throwing something against something"     → impacto con transmisión de momento
  - 153: "Throwing something in the air and catching it" → velocidad ascendente + descendente
  - 154: "Throwing something in the air and letting it fall" → lanzamiento + caída libre
  - 155: "Throwing something onto a surface"        → proyección dirigida
  - 139: "Spinning something so it continues spinning" → velocidad angular sostenida

BAJA VELOCIDAD (speed_label=0):
  - 134: "Something falling like a feather or paper" → caída amortiguada por resistencia del aire
  - 55:  "Poking something so it slightly moves"     → fuerza mínima, desplazamiento mínimo
  - 56:  "Poking something so lightly that it doesn't or almost doesn't move" → casi estático
  - 160: "Touching (without moving) part of something" → estático, sin movimiento
  - 16:  "Holding something"                         → objeto en reposo relativo
  - 100: "Pushing something so that it slightly moves" → desplazamiento mínimo controlado
  - 140: "Spinning something that quickly stops spinning" → velocidad angular decreciente rápida

CONTEO ESPERADO:
  - SSv2 train tiene ~168,913 videos totales con 174 clases
  - Cada clase tiene ~970 videos en promedio
  - 7 categorías rápidas × ~970 ≈ 6,790 videos rápidos
  - 7 categorías lentas × ~970 ≈ 6,790 videos lentos
  - Total esperado: ~13,580 videos con buen balance 50/50

USO:
  python create_physical_subset.py [--data_root RUTA] [--output RUTA_CSV]
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ============================================================================
# Definición de categorías como proxy de velocidad
# Cada entrada es (class_id, nombre, justificación breve)
# ============================================================================
FAST_CATEGORIES = {
    135: "Something falling like a rock",
    151: "Throwing something",
    152: "Throwing something against something",
    153: "Throwing something in the air and catching it",
    154: "Throwing something in the air and letting it fall",
    155: "Throwing something onto a surface",
    139: "Spinning something so it continues spinning",
}

SLOW_CATEGORIES = {
    134: "Something falling like a feather or paper",
    55:  "Poking something so it slightly moves",
    56:  "Poking something so lightly that it doesn't or almost doesn't move",
    160: "Touching (without moving) part of something",
    16:  "Holding something",
    100: "Pushing something so that it slightly moves",
    140: "Spinning something that quickly stops spinning",
}


def load_labels_map(labels_path: str) -> dict:
    """
    Carga labels.json y retorna un diccionario {class_id (int): nombre (str)}.

    labels.json tiene formato: {"nombre_clase": "class_id_str", ...}
    Lo invertimos a: {class_id_int: "nombre_clase"}
    """
    with open(labels_path, "r") as f:
        raw = json.load(f)
    # raw: {"Approaching something with your camera": "0", ...}
    # Invertimos: {0: "Approaching something with your camera", ...}
    return {int(v): k for k, v in raw.items()}


def parse_ssv2_csv(csv_path: str) -> list:
    """
    Parsea un CSV de SSv2 con formato: 'path num_frames class_id'
    (separado por espacios, NO por comas).

    Retorna lista de tuplas: [(video_path, num_frames, class_id), ...]
    """
    entries = []
    with open(csv_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 2)  # split desde la derecha para manejar paths con espacios
            if len(parts) != 3:
                print(f"  [ADVERTENCIA] Línea {line_num} tiene formato inesperado: '{line}'")
                continue
            video_path, num_frames_str, class_id_str = parts
            try:
                entries.append((video_path, int(num_frames_str), int(class_id_str)))
            except ValueError:
                print(f"  [ADVERTENCIA] Línea {line_num} tiene valores no numéricos: '{line}'")
                continue
    return entries


def create_physical_subset(entries: list, labels_map: dict) -> list:
    """
    Filtra las entradas cuyo class_id pertenezca a las categorías de velocidad.

    Retorna lista de diccionarios con columnas:
      video_path, num_frames, original_label, class_id, speed_label
    """
    all_speed_ids = set(FAST_CATEGORIES.keys()) | set(SLOW_CATEGORIES.keys())
    subset = []
    for video_path, num_frames, class_id in entries:
        if class_id not in all_speed_ids:
            continue
        speed_label = 1 if class_id in FAST_CATEGORIES else 0
        original_label = labels_map.get(class_id, f"unknown_{class_id}")
        subset.append({
            "video_path": video_path,
            "num_frames": num_frames,
            "original_label": original_label,
            "class_id": class_id,
            "speed_label": speed_label,
        })
    return subset


def write_csv(subset: list, output_path: str):
    """
    Escribe el subconjunto diagnóstico como CSV estándar (separado por comas).
    """
    import csv
    fieldnames = ["video_path", "num_frames", "original_label", "class_id", "speed_label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)


def print_summary(subset: list):
    """Imprime un resumen del balance de clases."""
    fast_count = sum(1 for s in subset if s["speed_label"] == 1)
    slow_count = sum(1 for s in subset if s["speed_label"] == 0)
    total = len(subset)

    print("\n" + "=" * 60)
    print("RESUMEN DEL SUBCONJUNTO DIAGNÓSTICO DE VELOCIDAD")
    print("=" * 60)
    print(f"  Total de videos seleccionados: {total}")
    print(f"  Videos RÁPIDOS (speed_label=1): {fast_count} ({100*fast_count/max(total,1):.1f}%)")
    print(f"  Videos LENTOS  (speed_label=0): {slow_count} ({100*slow_count/max(total,1):.1f}%)")
    print(f"  Balance ratio: {min(fast_count, slow_count) / max(max(fast_count, slow_count), 1):.2f}")

    print("\n  Desglose por categoría (RÁPIDO):")
    fast_by_cat = {}
    slow_by_cat = {}
    for s in subset:
        cid = s["class_id"]
        if s["speed_label"] == 1:
            fast_by_cat[cid] = fast_by_cat.get(cid, 0) + 1
        else:
            slow_by_cat[cid] = slow_by_cat.get(cid, 0) + 1

    for cid, name in sorted(FAST_CATEGORIES.items()):
        count = fast_by_cat.get(cid, 0)
        print(f"    [{cid:3d}] {name}: {count} videos")

    print("\n  Desglose por categoría (LENTO):")
    for cid, name in sorted(SLOW_CATEGORIES.items()):
        count = slow_by_cat.get(cid, 0)
        print(f"    [{cid:3d}] {name}: {count} videos")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Genera physical_diagnostics.csv a partir de SSv2 para estudiar velocidad"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/ssv2_luis",
        help="Ruta raíz del dataset SSv2 (contiene labels/ y SomethingV2/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="physical_diagnostics.csv",
        help="Ruta de salida para el CSV diagnóstico",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Splits a incluir (default: train val)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # 1. Cargar mapa de etiquetas
    labels_path = data_root / "labels" / "labels" / "labels.json"
    if not labels_path.exists():
        print(f"ERROR: No se encontró {labels_path}")
        sys.exit(1)
    print(f"Cargando etiquetas desde: {labels_path}")
    labels_map = load_labels_map(str(labels_path))
    print(f"  → {len(labels_map)} clases encontradas")

    # Verificar que nuestras categorías de velocidad existen en labels_map
    all_speed_ids = set(FAST_CATEGORIES.keys()) | set(SLOW_CATEGORIES.keys())
    missing = all_speed_ids - set(labels_map.keys())
    if missing:
        print(f"ADVERTENCIA: Las siguientes class_ids no existen en labels.json: {missing}")

    # 2. Parsear CSVs de los splits
    all_entries = []
    for split in args.splits:
        csv_path = data_root / "labels" / "sthv2" / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} no encontrado")
            continue
        print(f"Parseando {csv_path}...")
        entries = parse_ssv2_csv(str(csv_path))
        print(f"  → {len(entries)} entradas en {split}")
        all_entries.extend(entries)

    if not all_entries:
        print("ERROR: No se encontraron entradas en ningún split.")
        sys.exit(1)

    # 3. Filtrar por categorías de velocidad
    print(f"\nFiltrando por {len(all_speed_ids)} categorías de velocidad...")
    subset = create_physical_subset(all_entries, labels_map)

    if not subset:
        print("ERROR: No se encontraron videos que coincidan con las categorías de velocidad.")
        sys.exit(1)

    # 4. Escribir CSV
    write_csv(subset, args.output)
    print(f"\n✅ CSV generado: {args.output}")

    # 5. Resumen
    print_summary(subset)


if __name__ == "__main__":
    main()
