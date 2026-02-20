import pandas as pd
import os
import subprocess

# Configuración
csv_file = 'labels/val.csv'  # Tu archivo de validación
output_folder = 'k400'       # La carpeta destino
num_videos = 5               # Descargar solo 5 para probar

# Crear carpeta si no existe
os.makedirs(output_folder, exist_ok=True)

# Leer el CSV (sin cabecera, columna 0 es la ruta)
df = pd.read_csv(csv_file, header=None, sep=' ')
print(f"Total videos en CSV: {len(df)}")

# Intentar descargar los primeros N videos
count = 0
for index, row in df.iterrows():
    if count >= num_videos:
        break
    
    # Ruta: k400/video_id.mp4 -> Extraemos el ID
    rel_path = row[0]  
    video_filename = os.path.basename(rel_path) 
    video_id = os.path.splitext(video_filename)[0] 
    
    output_path = os.path.join(output_folder, video_filename)
    
    if os.path.exists(output_path):
        print(f"[YA EXISTE] {output_path}")
        count += 1
        continue

    print(f"Descargando {video_id} ...")
    
    # URL de YouTube
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Comando yt-dlp (Descarga en MP4, max 360p)
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]/mp4",
        "-o", output_path,
        url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[EXITO] Guardado en {output_path}")
        count += 1
    except subprocess.CalledProcessError:
        print(f"[ERROR] No se pudo descargar {video_id} (quizás borrado de YT)")

print("--- Proceso de muestra terminado ---")
