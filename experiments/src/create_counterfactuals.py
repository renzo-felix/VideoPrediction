import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from config import *

def create_counterfactual(video_path, output_dir_fast, output_dir_slow, video_id):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionamos la imagen para estandarizar el input a los modelos (ej. 224x224)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
        
    cap.release()
    
    # Necesitamos suficientes frames para poder submuestrear (ej. queremos generar NUM_FRAMES=16)
    # Por lo tanto, necesitamos al menos NUM_FRAMES * 2 frames en el video original.
    if len(frames) < NUM_FRAMES * 2:
        return False 
        
    # 1. CLON RÁPIDO: Sub-muestreo (nos saltamos 1 frame). La accion durará la mitad.
    fast_frames = frames[::2]
    
    # 2. CLON LENTO: Duplicamos cada frame. La acción durará el doble.
    slow_frames = []
    for f in frames:
        slow_frames.extend([f, f])
        
    # Recortamos ambos a exactamente NUM_FRAMES para el input del modelo (generalmente 16 frames)
    fast_frames = fast_frames[:NUM_FRAMES]
    slow_frames = slow_frames[:NUM_FRAMES]
    
    if len(fast_frames) < NUM_FRAMES:
        return False
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Guardamos el clon rápido
    fast_path = output_dir_fast / f"{video_id}_fast.mp4"
    out_fast = cv2.VideoWriter(str(fast_path), fourcc, FPS, (IMG_SIZE, IMG_SIZE))
    for f in fast_frames:
        out_fast.write(f)
    out_fast.release()
    
    # Guardamos el clon lento
    slow_path = output_dir_slow / f"{video_id}_slow.mp4"
    out_slow = cv2.VideoWriter(str(slow_path), fourcc, FPS, (IMG_SIZE, IMG_SIZE))
    for f in slow_frames:
        out_slow.write(f)
    out_slow.release()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generar pares contrafactuales de velocidad a partir de videos reales.")
    parser.add_argument("--input_dir", required=True, help="Ruta a la carpeta original de videos (ej. SSv2)")
    parser.add_argument("--num_videos", type=int, default=100, help="Cantidad de videos base a utilizar")
    parser.add_argument("--dataset_name", default="dataset", help="Nombre del dataset para diferenciar salidas (ej. ssv2 o k400)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    
    out_fast = VIDEOS_DIR / "counterfactuals" / args.dataset_name / "fast"
    out_slow = VIDEOS_DIR / "counterfactuals" / args.dataset_name / "slow"
    out_fast.mkdir(parents=True, exist_ok=True)
    out_slow.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_dir.glob("*.webm"))
    if not video_files:
        video_files = list(input_dir.glob("*.mp4"))
        
    if not video_files:
        print(f"Error: No se encontraron videos (.webm o .mp4) en {input_dir}")
        return
        
    print(f"Encontrados {len(video_files)} videos en la ruta origen.")
    
    # Selección aleatoria para variedad
    np.random.seed(42)
    selected_files = np.random.choice(video_files, min(args.num_videos * 3, len(video_files)), replace=False)
    
    metadata = []
    count = 0
    
    for vf in tqdm(selected_files, desc="Generando pares contrafactuales"):
        video_id = vf.stem
        success = create_counterfactual(vf, out_fast, out_slow, video_id)
        
        if success:
            # Registrar metadatos para el video rápido
            metadata.append({
                "video_id": f"{video_id}_fast",
                "original_id": video_id,
                "speed_group": "fast",
                "video_path": str(out_fast / f"{video_id}_fast.mp4"),
                "target_speed": 2.0 
            })
            
            # Registrar metadatos para el video lento
            metadata.append({
                "video_id": f"{video_id}_slow",
                "original_id": video_id,
                "speed_group": "slow",
                "video_path": str(out_slow / f"{video_id}_slow.mp4"),
                "target_speed": 0.5 
            })
            
            count += 1
            if count >= args.num_videos:
                break
                
    df = pd.DataFrame(metadata)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"counterfactual_metadata_{args.dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Éxito. Se generaron {count} pares ({count*2} videos en total).")
    print(f"✓ Metadatos guardados en: {csv_path}")

if __name__ == "__main__":
    main()
