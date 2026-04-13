### Misión de Investigación: Abstracción Arquitectónica e Inyección Dinámica de Hooks en V-JEPA2 y VideoMAEv2 (PLANTILLA)

> **⏸️ ESTADO: En standby** — Esperando que el compañero de equipo descargue los checkpoints de V-JEPA2 y confirme la estructura del repositorio. Los campos marcados con `[⚠️ COMPLETAR]` deben ser rellenados antes de ejecutar este prompt.

**Rol y Audiencia:**
Actúa como un Arquitecto de Deep Learning Senior y experto en Interpretabilidad Mecanicista de Transformers. El código que generes será utilizado por un equipo de investigadores que evalúa si los modelos fundacionales de video aprenden propiedades físicas del mundo real (velocidad) de manera universal, ejecutando los experimentos en el clúster HPC Khipu (nodos con RTX A6000 de 48GB VRAM).

**Contexto Científico y Operativo:**
Hemos comprobado que VideoMAEv2 decodifica la "velocidad" de forma robusta en su MLP final (96.5% accuracy en la Capa 39) sobre SSv2. Nuestra meta es expandir el pipeline para evaluar **V-JEPA2** y probar si esta cristalización semántica se replica en arquitecturas *Joint-Embedding*.

Nuestro script de extracción actual (`mechanistic_hooks.py`) está diseñado para VideoMAEv2, pero tras inspección del código fuente hemos determinado que:

**HALLAZGO CLAVE (Verificado en código fuente):**
Ambos modelos comparten la misma nomenclatura de submódulos:
- Bloques de atención: `model.blocks` (nn.ModuleList) — en AMBOS modelos
- Submódulo de atención: `model.blocks[i].attn` — en AMBOS modelos
- Submódulo MLP: `model.blocks[i].mlp` — en AMBOS modelos

**Diferencias técnicas reales entre las arquitecturas:**

| Propiedad | VideoMAEv2 (`models/modeling_finetune.py`) | V-JEPA2 (`vjepa2/src/models/vision_transformer.py` + `vjepa2/src/models/utils/modules.py`) |
|:---|:---|:---|
| Ruta a bloques | `model.blocks` | `model.blocks` ✅ idéntico |
| Ruta a atención | `model.blocks[i].attn` | `model.blocks[i].attn` ✅ idéntico |
| Ruta a MLP | `model.blocks[i].mlp` | `model.blocks[i].mlp` ✅ idéntico |
| Tipo atención | `CosAttention` (escalado learnable `nn.Parameter`) | `RoPEAttention` (Rotary Position Embedding) o `Attention` (dot-product + SDPA) |
| Tipo MLP | `Mlp` (fc1→GELU→fc2) | `MLP` (fc1→GELU→fc2) o `SwiGLUFFN` (fc1,fc2→SiLU→fc3) |
| LayerScale | Sí (`gamma_1`, `gamma_2`) | No |
| Forward del bloque | `forward(self, x)` — solo tensor de entrada | `forward(self, x, mask, attn_mask, T, H_patches, W_patches)` — parámetros posicionales extra |
| embed_dim | 1408 (fijo, ViT-Giant) | 1024 (ViT-L), 1280 (ViT-H), 1408 (ViT-G) |
| depth | 40 (fijo) | 24 (ViT-L), 32 (ViT-H), 40 (ViT-G) |
| Carga de modelo | `vit_giant_patch14_224(...)` + `load_state_dict(checkpoint)` | `vjepa2_vit_giant()` desde `src/hub/backbones.py` que retorna `(encoder, predictor)`. **Solo usar `encoder`.** |
| Checkpoint | `.pth` estándar | Retorna dict con keys `"encoder"` y `"predictor"` |

**Variantes disponibles de V-JEPA2 (verificadas en `src/hub/backbones.py`):**
- `vjepa2_vit_large`: depth=24, embed_dim=1024, img_size=256
- `vjepa2_vit_huge`: depth=32, embed_dim=1280, img_size=256
- `vjepa2_vit_giant`: depth=40, embed_dim=1408, img_size=256
- `vjepa2_vit_giant_384`: depth=40, embed_dim=1408, img_size=384

---

**[⚠️ COMPLETAR ANTES DE EJECUTAR] Pre-requisitos pendientes:**

1. **Checkpoint V-JEPA2:** Descargar a la ruta compartida del clúster:
   ```bash
   # [⚠️ COMPLETAR] Descargar el checkpoint de la variante elegida.
   # Ejemplo para ViT-Giant:
   wget -P /home/projects/video-prediction/checkpoints/vjepa2/ \
     https://dl.fbaipublicfiles.com/vjepa2/vitg.pt
   # Verificar: ls -la /home/projects/video-prediction/checkpoints/vjepa2/
   # El directorio actualmente está VACÍO.
   ```
   **¿Por qué?** Sin el checkpoint, no podemos cargar los pesos preentrenados del encoder. El archivo `vitg.pt` contiene un diccionario con keys `"encoder"` y `"predictor"` (ver `src/hub/backbones.py` líneas 136-140).

2. **Variante del modelo:** 
   ```
   # [⚠️ COMPLETAR] ¿Qué variante de V-JEPA2 usará el equipo?
   # Opciones: vit_large (depth=24), vit_huge (depth=32), vit_giant (depth=40)
   # Recomendación: vit_giant para comparación directa con VideoMAEv2 (ambos depth=40, embed_dim=1408)
   VJEPA2_VARIANT = "vit_giant"  # [⚠️ COMPLETAR]
   ```
   **¿Por qué?** La variante determina el embed_dim y depth. Para una comparación justa con VideoMAEv2, vit_giant (1408 dims, 40 capas) es la opción directa.

3. **Entorno conda de V-JEPA2:**
   ```
   # [⚠️ COMPLETAR] ¿Se usa el mismo env `videomae_luis_izaguirre` o uno diferente?
   # V-JEPA2 requiere: torch, timm, einops (ver vjepa2/requirements.txt)
   CONDA_ENV = "videomae_luis_izaguirre"  # [⚠️ COMPLETAR]
   ```
   **¿Por qué?** V-JEPA2 depende de `timm` y `einops`. Si el env actual no los tiene, se necesita instalarlos o crear un env separado.

---

**Tu Tarea Exclusiva:**
Refactoriza `mechanistic_hooks.py` creando la clase `UniversalVideoActivationExtractor` capaz de inyectar Forward Hooks automáticamente tanto para VideoMAEv2 como V-JEPA2.

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **Resolución Topológica Dinámica:**
   Dado que ambos modelos usan `model.blocks[i]`, `block.attn` y `block.mlp`, la resolución es directa:
   ```python
   num_blocks = len(model.blocks)  # 40 para ViT-Giant en ambos
   embed_dim = model.embed_dim     # Ambos modelos exponen este atributo
   ```
   Registrar hooks en `model.blocks[i]`, `model.blocks[i].attn`, `model.blocks[i].mlp`.

2. **Invarianza Dimensional Automática:**
   No hardcodear `1408`. Capturar la dimensión del tensor en tiempo de ejecución:
   ```python
   # El hook captura output.shape = [B, N, embed_dim]
   # embed_dim se obtiene de model.embed_dim al momento de inicializar
   ```
   Esto permite que el mismo extractor funcione con ViT-Large (1024), ViT-Huge (1280) y ViT-Giant (1408).

3. **Factory Method para Carga de Modelos:**
   ```python
   @staticmethod
   def load_model(model_name: str, checkpoint_path: str, device, **kwargs):
       if model_name == "videomaev2":
           from models import vit_giant_patch14_224
           model = vit_giant_patch14_224(num_classes=..., ...)
           # cargar checkpoint con load_state_dict
           return model
       elif model_name == "vjepa2":
           # [⚠️ COMPLETAR] Adaptar import path según estructura final de vjepa2
           # Opción A: usar torch.hub
           # encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
           # Opción B: importar directamente desde src/hub/backbones.py
           # from vjepa2.src.hub.backbones import vjepa2_vit_giant
           # encoder, predictor = vjepa2_vit_giant(pretrained=True)
           return encoder  # Solo el encoder, NO el predictor
   ```
   **¿Por qué completar?** La ruta de importación depende de cómo el compañero configure el proyecto. Si trabaja desde dentro de `vjepa2/`, los imports son diferentes a si lo importa desde la raíz `VideoPrediction/`.

4. **Preservación Estricta de Gestión de Memoria (Anti-OOM):**
   Mantener la arquitectura actual de mean pooling post-hook:
   - `.detach().cpu()` en hooks
   - `gc.collect()` + `torch.cuda.empty_cache()` en `clear_activations()`
   - No hacer mean pooling dentro del hook

5. **Comentarios de Conciliación de Grafos:**
   Docstring a nivel de clase explicando:
   - VideoMAEv2: CosAttention con escalado learnable + LayerScale (gamma_1/gamma_2)
   - V-JEPA2: RoPEAttention con Rotary Position Embedding, sin LayerScale
   - Los hooks capturan la salida post-módulo en ambos casos, la diferencia interna de atención no afecta la captura

**Entregable Esperado:**
El `mechanistic_hooks.py` actualizado con la clase universal, tipado estricto (Type Hints), y los `[⚠️ COMPLETAR]` claramente marcados para que el compañero los pueda rellenar.