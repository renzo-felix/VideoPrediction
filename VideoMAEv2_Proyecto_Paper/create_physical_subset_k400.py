"""
create_physical_subset_k400.py
==============================
Genera un CSV con un subconjunto diagnóstico del dataset Kinetics-400 (K400)
para estudiar la propiedad física de VELOCIDAD en VideoMAEv2 (ViT-Giant).

JUSTIFICACIÓN ARQUITECTÓNICA:
-----------------------------
Para el dataset K400, utilizamos pares de categorías como PROXY de velocidad,
siguiendo la misma metodología validada en SSv2 (Cristalización Semántica).
Seleccionamos clases basadas en el nivel de energía cinética, velocidad 
translacional o angular observada en la acción.

PARES DE CATEGORÍAS SELECCIONADOS (basados en K710 IDs limitados a <400):
--------------------------------------------------------------------------
ALTA VELOCIDAD (speed_label=1): Acciones con extrema energía o aceleración.
BAJA VELOCIDAD (speed_label=0): Acciones controladas, estáticas o manipulaciones lentas.

Se han excluido intencionalmente clases de K710 que poseen un ID >= 400 
(ej. calligraphy, ID=515) dado que no están presentes en el subset de K400.

USO:
  python create_physical_subset_k400.py --csv dataset/k400_luis/labels/val.csv --output physical_diagnostics_k400.csv
"""

import argparse
import csv
import os
import sys

# Intentar importar decord para contar los frames de los MP4
try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except ImportError:
    print("[ADVERTENCIA] decord no está instalado. Usaremos cv2 como alternativa.")
    DECORD_AVAILABLE = False
    import cv2


# ============================================================================
# Mapeo Antagonista (Proxy Físico)
# ============================================================================
FAST_CATEGORIES = {
    11: "bobsledding",           # Velocidad translacional extrema
    22: "pole vault",            # Aceleración + desplazamiento vertical rápido
    28: "skateboarding",         # Velocidad translacional sostenida
    29: "dunking basketball",    # Salto + impacto de alta energía
    46: "ski jumping",           # Velocidad extrema + vuelo
    54: "kicking field goal",    # Transferencia de energía cinética al balón
    132: "slapping",             # Velocidad de impacto manual
    155: "throwing discus",      # Lanzamiento atlético con momento angular
    222: "shot put",             # Lanzamiento pesado con fuerza máxima
    267: "hammer throw",         # Velocidad angular + lanzamiento
    295: "javelin throw",        # Proyección de alta velocidad
    299: "long jump",            # Sprint + salto horizontal
    300: "parkour",              # Movimiento acrobático rápido y continuo
}

SLOW_CATEGORIES = {
    37: "stretching leg",        # Movimiento lento, controlado
    123: "folding clothes",      # Manipulación manual lenta
    129: "tai chi",              # Movimiento deliberadamente lento
    219: "ironing",              # Desplazamiento mínimo y repetitivo
    248: "knitting",             # Casi estático, movimiento fino de dedos
    249: "reading book",         # Estático, sin movimiento significativo
    262: "stretching arm",       # Movimiento lento, controlado
    290: "folding napkins",      # Manipulación manual mínima
    311: "playing chess",        # Estático, movimiento mínimo
    319: "arranging flowers",    # Manipulación cuidadosa y lenta
    331: "watering plants",      # Movimiento suave, sin prisa
    371: "yoga",                 # Posturas estáticas o transiciones lentas
}

def get_video_frame_count(video_path: str) -> int:
    """
    Obtiene el número de frames de un video MP4.
    Usa decord si está disponible, sino recae en cv2.
    """
    if not os.path.exists(video_path):
        return 0

    if DECORD_AVAILABLE:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            return len(vr)
        except Exception as e:
            print(f"Error leyendo {video_path} con decord: {e}")
            return 0
    else:
        try:
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return length
        except Exception as e:
            print(f"Error leyendo {video_path} con cv2: {e}")
            return 0


def load_k710_labels(label_map_path: str) -> dict:
    """
    Lee label_map_k710.txt y devuelve el mapeo {class_id: class_name}.
    La línea 0 corresponde al class_id 0.
    """
    labels = {}
    with open(label_map_path, 'r') as f:
        for idx, line in enumerate(f):
            labels[idx] = line.strip()
    return labels


def parse_k400_csv(csv_path: str) -> list:
    """
    Parsea un CSV de K400 con formato: 'k400/NOMBRE_VIDEO.mp4 CLASS_ID'
    Retorna lista de tuplas: [(video_path, class_id), ...]
    """
    entries = []
    with open(csv_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                print(f"  [ADVERTENCIA] Línea {line_num} ignorada por formato inválido: '{line}'")
                continue
            video_path, class_id_str = parts
            try:
                entries.append((video_path, int(class_id_str)))
            except ValueError:
                print(f"  [ADVERTENCIA] Línea {line_num} ignorada por class_id inválido.")
                continue
    return entries


def main():
    parser = argparse.ArgumentParser(description="Genera physical_diagnostics_k400.csv")
    parser.add_argument("--csv", type=str, default="dataset/k400_luis/labels/val.csv",
                        help="Ruta al CSV de validación de K400")
    parser.add_argument("--label_map", type=str, default="label_map_k710.txt",
                        help="Ruta al mapa de clases de K710")
    parser.add_argument("--output", type=str, default="physical_diagnostics_k400.csv",
                        help="Archivo CSV de salida")
    args = parser.parse_args()

    # 1. Cargar el mapa de labels
    if not os.path.exists(args.label_map):
        print(f"ERROR: No se encontró el mapa de clases {args.label_map}")
        sys.exit(1)
    
    k710_labels = load_k710_labels(args.label_map)
    print(f"Mapa de K710 cargado con {len(k710_labels)} clases.")

    # 2. Cargar entradas del CSV de K400
    if not os.path.exists(args.csv):
        print(f"ERROR: No se encontró el dataset CSV en {args.csv}")
        sys.exit(1)
    
    entries = parse_k400_csv(args.csv)
    print(f"Parseadas {len(entries)} entradas desde {args.csv}.")

    # 3. Filtrar por los proxies de velocidad
    # Verificamos que los IDs sean válidos y mapeen a velocidad
    subset = []
    all_speed_ids = set(FAST_CATEGORIES.keys()) | set(SLOW_CATEGORIES.keys())
    
    missing_videos = 0
    corrupt_videos = 0

    print("Procesando y contando frames para videos filtrados. Esto puede tomar unos minutos...")
    for video_path, class_id in entries:
        if class_id not in all_speed_ids:
            continue
        
        # Validar existencia y número de frames
        full_path = os.path.join(".", video_path) # k400/...
        if not os.path.exists(full_path):
            missing_videos += 1
            continue
            
        num_frames = get_video_frame_count(full_path)
        if num_frames <= 0:
            corrupt_videos += 1
            continue

        speed_label = 1 if class_id in FAST_CATEGORIES else 0
        original_label = k710_labels.get(class_id, f"unknown_{class_id}")
        
        subset.append({
            "video_path": video_path,
            "num_frames": num_frames,
            "original_label": original_label,
            "class_id": class_id,
            "speed_label": speed_label,
        })

    # 4. Escribir resultado
    fieldnames = ["video_path", "num_frames", "original_label", "class_id", "speed_label"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)

    # 5. Imprimir resumen
    fast_count = sum(1 for s in subset if s["speed_label"] == 1)
    slow_count = sum(1 for s in subset if s["speed_label"] == 0)
    total = len(subset)

    print("\n" + "=" * 60)
    print("RESUMEN DEL SUBCONJUNTO DIAGNÓSTICO K400")
    print("=" * 60)
    print(f"  Videos faltantes ignorados: {missing_videos}")
    print(f"  Videos corruptos (0 frames) ignorados: {corrupt_videos}")
    print(f"  Total de videos válidos exportados: {total}")
    print(f"  Videos RÁPIDOS (speed_label=1): {fast_count}")
    print(f"  Videos LENTOS  (speed_label=0): {slow_count}")
    print(f"✅ CSV guardado exitosamente en: {args.output}")

if __name__ == "__main__":
    main()
