import numpy as np
import os

# Nombre del archivo que generaste
file_path = 'features_extraidos.npy'

print(f"📂 Analizando archivo: {file_path}...")

if not os.path.exists(file_path):
    print("❌ Error: No encuentro el archivo .npy. Verifica que estás en la carpeta correcta.")
    exit()

try:
    # Cargar el array
    data = np.load(file_path)

    print("\n" + "="*40)
    print("   RADIOGRAFÍA DE TUS FEATURES")
    print("="*40)
    
    # 1. Dimensiones (Lo más importante)
    print(f"📐 SHAPE (Dimensiones): {data.shape}")
    print(f"🔢 Cantidad total de valores: {data.size}")
    print(f"💾 Tipo de dato: {data.dtype}")
    
    # 2. Estadísticas rápidas (para ver si no está vacío)
    print("-" * 40)
    print(f"⬇️  Valor Mínimo: {np.min(data):.4f}")
    print(f"⬆️  Valor Máximo: {np.max(data):.4f}")
    print(f"Ø  Promedio:     {np.mean(data):.4f}")
    print("-" * 40)

    # 3. Muestra de los primeros valores
    print("👀 Primeros 10 valores (del vector aplanado):")
    print(data.flatten()[:10])
    print("="*40 + "\n")

    # Interpretación automática según la forma
    shape = data.shape
    print("💡 INTERPRETACIÓN:")
    if len(shape) == 2: # Ej: (1, 768)
        print("   Tienes un vector GLOBAL (CLS token).")
        print("   Esto representa el resumen de TODO el video en un solo vector.")
    elif len(shape) == 3: # Ej: (1, 1568, 768)
        print("   Tienes un mapa de PARCHES (Spatial/Temporal tokens).")
        print(f"   El video fue dividido en {shape[1]} cubos (tokens).")
        print("   Cada cubo tiene su propio vector de características.")
    
except Exception as e:
    print(f"❌ Error leyendo el archivo: {e}")