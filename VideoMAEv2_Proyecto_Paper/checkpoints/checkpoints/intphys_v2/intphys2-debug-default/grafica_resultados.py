import torch
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
ruta_archivo = "/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/checkpoints/checkpoints/intphys_v2/intphys2-debug-default/losses_10fs_4_6_8_10_12_14ctxt.pth"
contextos = [4, 6, 8, 10, 12, 14]  # Los contextos que definimos en el YAML

# --- CARGAR DATOS ---
try:
    data = torch.load(ruta_archivo, map_location='cpu', weights_only=False)
    losses = data['losses'] # Forma: [60, 6, 31]
    
    # 1. Promediar sobre los videos (dim 0) y los pasos (dim 2)
    # Nos quedará un array de 6 números (uno por cada contexto)
    loss_por_contexto = losses.mean(dim=(0, 2)).numpy()
    
    print("Loss por contexto:", loss_por_contexto)

    # --- GRAFICAR ---
    plt.figure(figsize=(10, 6))
    plt.plot(contextos, loss_por_contexto, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.title(f'VideoMAE v2 Giant: Sorpresa vs. Contexto\n(Promedio Global: {losses.mean():.4f})', fontsize=14)
    plt.xlabel('Cantidad de Frames de Contexto (Historia)', fontsize=12)
    plt.ylabel('Loss de Reconstrucción (Sorpresa)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar la imagen
    nombre_imagen = 'resultado_intphys_videomae.png'
    plt.savefig(nombre_imagen)
    print(f"\n✅ ¡Gráfica guardada como '{nombre_imagen}'!")
    print("Descárgala y ponla en tu presentación.")

except Exception as e:
    print(f"Error: {e}")