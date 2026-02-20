import torch
import numpy as np
import os
import argparse
import sys 

# --- SOLUCIÓN DEFINITIVA AL ERROR DE IMPORTACIÓN ---
# Definimos la ruta base del proyecto
project_path = '/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper'

# Definimos la ruta de la carpeta 'models' donde se esconde el archivo
models_path = os.path.join(project_path, 'models')

# Le decimos a Python: "Busca en la raíz Y TAMBIÉN en la carpeta models"
if project_path not in sys.path:
    sys.path.append(project_path)

if models_path not in sys.path:
    sys.path.append(models_path) # <--- ESTA ES LA LÍNEA MÁGICA
# ----------------------------------------

# AHORA sí encontrará 'modeling_finetune' porque ya sabe buscar en 'models/'
try:
    from modeling_finetune import vit_base_patch16_224
except ImportError:
    # Plan B: Por si el archivo se llama diferente o la estructura es compleja
    try:
        from models.modeling_finetune import vit_base_patch16_224
    except ImportError as e:
        print(f"❌ Error crítico importando el modelo: {e}")
        print(f"Rutas de búsqueda actuales: {sys.path}")
        exit()

from torchvision import transforms
from PIL import Image

# --- CONFIGURACIÓN ---
# Usamos las mismas rutas que ya sabemos que funcionan
MODEL_PATH = '/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/checkpoints/checkpoints/vit_b_k710_dl_from_giant.pth'
VIDEO_PATH = '/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/data/k710/video_test.mp4'
SAVE_PATH = 'features_extraidos.npy'

def get_args():
    parser = argparse.ArgumentParser()
    # Argumentos dummy para que no falle si el modelo los pide
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    print(f"🔄 Iniciando extracción de features...")
    
    # 1. Crear el modelo (Arquitectura Base)
    # Definimos num_classes=710 para que coincida con el archivo .pth, 
    # aunque luego le quitaremos la cabeza.
    print("   Creando modelo ViT-Base...")
    model = vit_base_patch16_224(num_classes=710)
    
    # 2. Cargar los Pesos (El Cerebro)
    print(f"   Cargando pesos desde: {os.path.basename(MODEL_PATH)}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Manejo de versiones de checkpoint (a veces vienen en 'module' o 'model')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint['module']
    
    # Limpiamos prefijos si fue entrenado en DDP
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_model.items()}
    
    # Cargamos y enviamos a GPU
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"   Estado de carga: {msg}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. "LOBOTOMÍA": Quitar la cabeza clasificadora
    # Reemplazamos la capa final (fc_norm y head) por una identidad
    # Así el modelo nos devolverá el vector de 768 dimensiones (CLS token)
    model.head = torch.nn.Identity()
    # model.fc_norm = torch.nn.Identity() # Opcional: a veces queremos features pre-norm

    # 4. Preparar el Video (Pre-procesamiento manual simplificado)
    # Como VideoMAE es complejo de cargar con dataloaders, usaremos Decord directo
    # para asegurar que vemos lo mismo que el modelo.
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    except ImportError:
        print("❌ Error: Necesitas instalar decord (pip install decord)")
        return

    # Extraemos 16 frames uniformemente (como en el entrenamiento)
    frame_indices = np.linspace(0, len(vr) - 1, 16).astype(int)
    buffer = vr.get_batch(frame_indices).asnumpy() # (16, H, W, C)

    # Transformaciones (Normalización ImageNet)
    # VideoMAE espera: (Batch, Channel, Time, Height, Width)
    inputs = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for frame in buffer:
        inputs.append(transform(frame))
    
    # Apilamos: (Time, C, H, W) -> (C, Time, H, W) -> Batch
    input_tensor = torch.stack(inputs).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    
    # 5. Inferencia (Extracción)
    print("   🧠 Extrayendo pensamientos (features)...")
    with torch.no_grad():
        # En algunos modelos ViT, forward_features da el CLS token + parches
        # forward_head (que ahora es identidad) nos dará lo que salía antes de clasificar.
        
        # Opción A: Usar forward_features (Más control)
        features = model.forward_features(input_tensor)
        
        # En VideoMAE, forward_features suele devolver (Batch, Time, Tokens, Dim)
        # o solo (Batch, Tokens, Dim) si hace pooling.
        # Si es un tensor estándar ViT, features suele ser el CLS token o la secuencia.
        
        # Vamos a guardar TODO lo que salga para analizarlo luego
        features_np = features.cpu().numpy()

    # 6. Guardar
    np.save(SAVE_PATH, features_np)
    print("\n" + "=" * 40)
    print(f"✅ ¡Éxito! Features guardados en: {SAVE_PATH}")
    print(f"📊 Dimensiones del vector: {features_np.shape}")
    print("=" * 40)
    print("Ahora puedes analizar este archivo .npy para ver activaciones.")

if __name__ == '__main__':
    main()