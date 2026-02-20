import torch
import numpy as np # Importante por si acaso

# --- TU RUTA DEL ARCHIVO ---
# Asegúrate de que esta sea la ruta correcta donde encontraste el archivo
ruta_archivo = "/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/checkpoints/checkpoints/intphys_v2/intphys2-debug-default/losses_10fs_4_6_8_10_12_14ctxt.pth"

try:
    # --- CORRECCIÓN AQUÍ: Agregamos weights_only=False ---
    data = torch.load(ruta_archivo, map_location='cpu', weights_only=False)
    # -----------------------------------------------------
    
    print("¡Archivo cargado con éxito! 🎉")
    print("-" * 30)
    print(f"Claves encontradas: {data.keys()}")
    
    # 1. Analizar los Losses (Sorpresa)
    if 'losses' in data:
        losses = data['losses']
        print(f"\n📊 Forma del tensor de Losses: {losses.shape}")
        # Forma típica: [N_videos, N_contextos, N_algo]
        
        print(f"Promedio Global de Sorpresa (Loss): {losses.mean().item():.4f}")
        print(f"Rango de Sorpresa: Min {losses.min().item():.4f} - Max {losses.max().item():.4f}")
    
    # 2. Analizar los Nombres
    if 'names' in data:
        nombres = data['names']
        print(f"\n🎥 Cantidad de videos evaluados: {len(nombres)}")
        print(f"Ejemplo de primeros 3 videos: {nombres[:3]}")
        
    # 3. Verificar alineación
    if 'losses' in data and 'names' in data:
        if len(losses) == len(nombres):
            print("\n✅ ¡La cantidad de scores coincide con la cantidad de videos!")
        else:
            print(f"\n⚠️ OJO: Hay {len(losses)} scores pero {len(nombres)} nombres.")

except FileNotFoundError:
    print("❌ No encontré el archivo. Revisa la ruta.")
except Exception as e:
    print(f"❌ Error al leer: {e}")