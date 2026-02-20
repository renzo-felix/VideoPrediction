import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. CORRECCIÓN DE RUTAS (Vital) ---
project_path = '/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper'
models_path = os.path.join(project_path, 'models')
if project_path not in sys.path: sys.path.append(project_path)
if models_path not in sys.path: sys.path.append(models_path)

# Importamos el modelo
try:
    from modeling_finetune import vit_base_patch16_224
except ImportError:
    from models.modeling_finetune import vit_base_patch16_224

from torchvision import transforms

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(project_path, 'checkpoints/checkpoints/vit_b_k710_dl_from_giant.pth')
VIDEO_PATH = os.path.join(project_path, 'data/k710/video_test.mp4')
OUTPUT_IMAGE = 'mapa_atencion_didgeridoo.jpg'

# Variable global para guardar la atención
attention_storage = []

def get_attention_hook(module, input, output):
    # El input de la capa Dropout de atención es la matriz de atención (Probabilidades)
    # input es una tupla, el elemento 0 es el tensor
    attention_storage.append(input[0].detach().cpu())

def main():
    print("🎨 Preparando visualización de atención...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar Modelo
    print("   Cargando modelo...")
    model = vit_base_patch16_224(num_classes=710)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Limpieza de pesos
    ckpt_model = checkpoint['model'] if 'model' in checkpoint else checkpoint['module']
    state_dict = {k.replace('module.', ''): v for k, v in ckpt_model.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. INSTALAR EL GANCHO (HOOK)
    # Vamos a espiar el último bloque del Transformer (block 11 para ViT-Base)
    # Específicamente la capa 'attn_drop' dentro de 'attn'.
    # Su entrada son los pesos de atención normalizados (Softmax).
    target_layer = model.blocks[-1].attn.attn_drop
    hook = target_layer.register_forward_hook(get_attention_hook)
    print("   🪝 Gancho instalado en el último bloque.")

    # 3. Cargar Video
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    except:
        print("❌ Error: Falta decord.")
        return

    # Muestrear 16 frames
    frame_indices = np.linspace(0, len(vr) - 1, 16).astype(int)
    frames_raw = vr.get_batch(frame_indices).asnumpy() # (16, H, W, C) RGB

    # Preprocesar para el modelo
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inputs = [transform(f) for f in frames_raw]
    input_tensor = torch.stack(inputs).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # 4. Inferencia
    print("   🧠 Ejecutando modelo...")
    with torch.no_grad():
        model(input_tensor)

# 5. Procesar Atención
    # attention_storage[0] tiene shape: (Batch, Heads, N_tokens, N_tokens)
    attn_map = attention_storage[0][0] # Tomamos el primer video
    print(f"   Matriz de atención capturada: {attn_map.shape}")

    # Promediamos las cabezas de atención (Heads)
    attn_mean = torch.mean(attn_map, dim=0) # Shape: (N_tokens, N_tokens)

    # --- CORRECCIÓN MATEMÁTICA INTELIGENTE ---
    # Total de parches esperados para (8, 14, 14)
    expected_patches = 8 * 14 * 14 # 1568
    
    current_tokens = attn_mean.shape[0]

    if current_tokens == expected_patches:
        print("   ℹ️  Modelo tipo GAP (Sin CLS token). Calculando atención global...")
        # Si tenemos exactamente 1568 tokens, no hay CLS que borrar.
        # Calculamos cuánto es "mirado" cada parche por todos los demás (promedio por columnas)
        # Esto nos da el mapa de importancia global.
        cls_attn = torch.mean(attn_mean, dim=0) # (1568,)
        
    elif current_tokens == expected_patches + 1:
        print("   ℹ️  Modelo con CLS token detectado.")
        # Si sobra 1, ese es el CLS (índice 0). Lo usamos como referencia y lo quitamos del mapa.
        cls_attn = attn_mean[0, 1:] # (1568,)
        
    else:
        print(f"❌ Error de dimensiones: Se tienen {current_tokens} tokens pero se esperaban {expected_patches}.")
        return
    # ------------------------------------------

    # Ahora sí, el reshape funcionará perfecto
    attn_grid = cls_attn.reshape(8, 14, 14).numpy()

    # 6. Generar Visualización
    print("   🖌️ Pintando mapas de calor...")
 
    # Tomamos un frame representativo (ej. el del medio)
    frame_idx = 8 
    tubelet_idx = frame_idx // 2 # Porque cada mapa de atención cubre 2 frames
    
    # Imagen original
    img_original = frames_raw[frame_idx]
    img_original = cv2.resize(img_original, (224, 224))
    
    # Mapa de atención correspondiente
    heatmap = attn_grid[tubelet_idx]
    
    # Normalizar heatmap (0 a 1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Redimensionar heatmap al tamaño de la imagen (224x224)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Convertir a colores (Jet colormap)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Superponer (Mezcla)
    result = cv2.addWeighted(img_original, 0.6, heatmap_color, 0.4, 0)
    
    # Guardar
    cv2.imwrite(OUTPUT_IMAGE, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*40)
    print(f"✅ ¡LISTO! Imagen guardada en: {OUTPUT_IMAGE}")
    print("="*40)
    print("Esta imagen muestra en ROJO lo que el modelo cree que es un Didgeridoo.")

if __name__ == '__main__':
    main()