**Contexto y Rol:**
Actúa como un Ingeniero de Machine Learning Senior especializado en **Interpretabilidad Mecanicista** y despliegue en clústeres de Alto Rendimiento (HPC). Nuestro objetivo es investigar en qué bloques y componentes (Attention vs. MLP) del modelo **VideoMAEv2 (ViT-Giant)** se codifican propiedades físicas fundamentales, específicamente la **velocidad**, utilizando el dataset Something-Something V2 (SSv2). Usaremos técnicas de **Linear Probing** y **Activation Patching** inspiradas en los notebooks `1_3_1_Linear_Probes_exercises.ipynb` y `part1_Transformer_from_Scratch_(exercises).ipynb`.

---

**Arquitectura Exacta del Modelo (ViT-Giant de `models/modeling_finetune.py`):**
- Función de registro: `vit_giant_patch14_224`
- `depth = 40` bloques (índices 0 a 39 inclusive)
- `embed_dim = 1408`
- `num_heads = 16`
- `patch_size = 14×14`, `tubelet_size = 2`
- Tipo de atención: `CosAttention` (escalado learnable, NO dot-product estándar)
- Escalado por `gamma_1`/`gamma_2` (LayerScale) en cada bloque
- **Forma del Residual Stream:** `[B, N, 1408]` donde `N = (T//2) × (H//14) × (W//14)`
  - Para entrada estándar `[B, 3, 16, 224, 224]`: `N = 8 × 16 × 16 = 2048` tokens

**Restricción Estricta de Código Base:**
Bajo ninguna circunstancia modifiques los archivos originales del modelo (`models/modeling_finetune.py` o `run_class_finetuning.py`). Toda extracción de activaciones y patching debe hacerse externamente usando **PyTorch Forward Hooks**.

---

**Plan de Trabajo a Implementar:**

**Script 1: `create_physical_subset.py`**
Procesa las etiquetas de `dataset/ssv2_luis/` y genera `physical_diagnostics.csv` con dos columnas de velocidad.
- **Alta velocidad:** etiquetas SSv2 que contengan "fast", "quickly", "suddenly" o similares.
- **Baja velocidad:** etiquetas que contengan "slowly", "carefully", "gently" o similares.
- El CSV debe incluir columnas: `video_path`, `original_label`, `speed_label` (0=slow, 1=fast).
- Documenta en el script qué etiquetas específicas de SSv2 se seleccionaron y por qué representan velocidad física.

**Script 2: `mechanistic_hooks.py`**
Clase `VideoMAEActivationExtractor` que envuelve el modelo ViT-Giant sin modificarlo.
- Usar `register_forward_hook` en `model.blocks[i]`, `model.blocks[i].attn` y `model.blocks[i].mlp`.
- Capas estratégicas por defecto: `[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]` (distribución uniforme sobre los 40 bloques).
- El hook debe almacenar el tensor de salida con forma `[B, N, 1408]` en un diccionario `{"block_{i}_residual": tensor, "block_{i}_attn": tensor, "block_{i}_mlp": tensor}`.
- Incluir método `clear_hooks()` para evitar memory leaks entre batches.
- **Reducción para probing:** aplicar `.mean(dim=1)` sobre los N tokens para obtener `[B, 1408]` por capa, antes de entrenar el clasificador lineal.

**Script 3: `run_layer_probing.py`**
Entrena clasificadores lineales (`sklearn.linear_model.LogisticRegression` o `torch.nn.Linear`) sobre las activaciones de cada capa/componente.
- Integrar **Weights & Biases (WandB)** para registrar:
  - `layer_probe_accuracy` por capa y por componente (Residual / Attn / MLP).
  - Gráfica de "Layer-wise Probe Accuracy" (eje X = índice de capa 0-39, eje Y = accuracy).
  - Matrices de confusión por capa.
- Procesar en batches pequeños (≤4) para no saturar la VRAM de 48GB del RTX A6000.
- Guardar los resultados en `output_dir/probing_results.json`.

**Script 4: `causal_patching_experiment.py`**
Implementa Activation Patching causal para verificar causalidad (no solo correlación).
- Toma un video A (rápido) y video B (lento) del CSV generado.
- Extrae la activación del bloque `layer_idx` de A: `h_A = activations["block_{layer_idx}_residual"]`.
- En la inferencia de B, reemplaza la activación en ese bloque con `h_A` (usando un pre-forward hook).
- Mide si la predicción de B cambia hacia la clase de A (evidencia causal de que esa capa codifica velocidad).
- Integrar WandB para registrar el "patching flip rate" por capa.

---

**Instrucciones Obligatorias de Ejecución en HPC (Khipu):**
Para cada script, genera también su archivo `.sh` de Slurm con este protocolo exacto:

```bash
#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=videomae_probing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Verificar disponibilidad del nodo antes de lanzar:
# sinfo -n ds001 -o "%N %t"

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# Solucionar fragmentación de memoria en ViT-Giant (40 bloques × 1408 dim)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Solucionar errores de compilación JIT de Triton (stdlib.h no encontrado)
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

CHECKPOINT="checkpoints/vit_g_hybrid_pt_1200e_k710_ft.pth"
```

---

**Estándares de Documentación del Código:**
- **Dimensiones de tensores:** Comenta la forma en cada paso crítico. Ejemplo:
  ```python
  # x: [B, N, 1408] donde N=2048 tokens (8 temporal × 256 espacial)
  # Después de mean pooling: [B, 1408] — representación global del video
  ```
- **Justificación arquitectónica:** En cada archivo, incluye un docstring explicando por qué ese punto de extracción es relevante. Por ejemplo, explica por qué el Residual Stream (salida del bloque completo) es mejor punto de partida para probing que solo el MLP Output.
- **Explicación de hooks:** Explica en comentarios cómo `register_forward_hook` captura activaciones sin mutar el grafo computacional original.

---

**Acción Inmediata:**
Comienza redactando `create_physical_subset.py` y su script Slurm `run_subset.sh`. En el docstring del script, documenta explícitamente:
1. Qué etiquetas de SSv2 seleccionas para "alta velocidad" y "baja velocidad".
2. Por qué esas etiquetas representan el concepto físico de velocidad (y no solo urgencia o fuerza).
3. Cuántos videos aproximados esperas en cada clase (para evaluar el balance del dataset).