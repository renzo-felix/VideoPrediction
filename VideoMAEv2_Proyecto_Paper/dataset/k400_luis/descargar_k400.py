import pandas as pd
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURACIÓN ---
CSV_FILE = 'labels/val.csv'
OUTPUT_FOLDER = 'k400'
FAILED_LOG = 'videos_fallidos.txt'
MAX_WORKERS = 8  # 8 descargas simultáneas

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def descargar_video(ruta_relativa):
    video_filename = os.path.basename(ruta_relativa)
    video_id = os.path.splitext(video_filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, video_filename)
    
    # Si ya existe, saltarlo
    if os.path.exists(output_path):
        return (video_id, True, "Ya existe")
        
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    cmd = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--socket-timeout", "10",
        "-f", "bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]/mp4",
        "-o", output_path,
        url
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (video_id, True, "Descargado")
    except subprocess.CalledProcessError:
        return (video_id, False, "No disponible (Borrado/Privado)")

def main():
    print(f"Leyendo {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, header=None, sep=' ')
    rutas = df[0].tolist()
    total = len(rutas)
    print(f"Total de videos a procesar: {total}")
    
    # Limpiar log anterior si existe
    if os.path.exists(FAILED_LOG):
        os.remove(FAILED_LOG)
        
    fallidos = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(descargar_video, ruta): ruta for ruta in rutas}
        
        completados = 0
        for futuro in as_completed(futuros):
            completados += 1
            video_id, exito, mensaje = futuro.result()
            
            if exito:
                print(f"[{completados}/{total}] [EXITO] {video_id} - {mensaje}")
            else:
                print(f"[{completados}/{total}] [ERROR] {video_id} - {mensaje}")
                fallidos += 1
                # Guardar el fallido en el txt inmediatamente
                with open(FAILED_LOG, 'a') as f:
                    f.write(f"{video_id}\n")

    print(f"\n--- Resumen Final ---")
    print(f"Total procesados: {total}")
    print(f"Fallidos: {fallidos}")
    print(f"Lista de fallidos guardada en: {FAILED_LOG}")

if __name__ == "__main__":
    main()