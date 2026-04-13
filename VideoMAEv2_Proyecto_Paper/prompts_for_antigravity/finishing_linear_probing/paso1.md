### Misión de Investigación: Refactorización Defensiva de Memoria para Extracción Exhaustiva Capa a Capa (VideoMAEv2)

**Rol y Audiencia:**
Actúa como un Ingeniero Senior de Machine Learning y MLOps especializado en Interpretabilidad Mecanicista, optimización de hardware a bajo nivel (PyTorch) y despliegue en entornos HPC (Slurm). Tu código será revisado por un equipo de investigadores analizando propiedades físicas (velocidad) en modelos fundacionales de video.

**Contexto y Estado Actual del Proyecto (Verificado):**
Nuestro script `mechanistic_hooks.py` (248 líneas) contiene la clase `VideoMAEActivationExtractor` que extrae representaciones de **11 capas discretas** (`DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]` en la línea 68) de un modelo ViT-Giant (`embed_dim=1408`, `depth=40`).

Para lograr una animación de clustering (PCA+UMAP) que muestre una evolución semántica fluida e ininterrumpida, debemos escalar la extracción a las **40 capas completas**. Esto generará 120 combinaciones de tensores por video (40 capas × 3 submódulos: Residual, MHA, MLP).

Operamos en el clúster Khipu utilizando nodos con GPU NVIDIA RTX A6000 (48GB VRAM).

**Referencia de tiempos reales (Job 29344):** La ejecución previa con 11 capas tardó 4h 21min en data-science. Con 40 capas (3.6× más), se estima ~16h de ejecución. El script Slurm debe solicitar tiempo suficiente.

**Archivos existentes que NO deben sobrescribirse:**
- `output_dir/activations.npz` (3.6 GB) — contiene las activaciones de 11 capas × 3 componentes × 21,202 videos
- `output_dir/probing_results.json` (7.5 KB) — resultados de linear probing de las 11 capas

**Logging a WandB:**
El script `run_layer_probing.py` ya tiene integración con Weights & Biases (líneas 345-404). Incluye:
- Función `log_to_wandb()` que sube accuracy por capa, confusion matrices y gráficas interactivas
- Argumentos CLI: `--wandb_project` (default: `videomae_probing`) y `--no_wandb` para desactivar
- La API key de WandB ya está configurada en `~/.netrc` del clúster
- **IMPORTANTE:** En el job 29344 hubo un error de autenticación WandB. Si vuelve a fallar, agregar en el script Slurm:
  ```bash
  export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
  ```
  O pasar `--no_wandb` para evitar que el job falle por WandB.

**Tu Tarea:**
Refactoriza `mechanistic_hooks.py` y actualiza `run_layer_probing.py` para soportar la extracción de las 40 capas completas, aplicando estrictas buenas prácticas de HPC. No debes modificar los archivos del modelo original (`models/modeling_finetune.py` — arquitectura caja negra).

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **Escalado de la lista de capas:**
   En `mechanistic_hooks.py` línea 68, cambiar:
   ```python
   DEFAULT_LAYERS = list(range(40))  # Las 40 capas completas
   ```
   Y en `run_layer_probing.py` línea 433, actualizar el default del argumento `--layers`:
   ```python
   parser.add_argument("--layers", type=int, nargs="+",
                       default=list(range(40)),
                       help="Capas a analizar (default: todas las 40 capas)")
   ```

2. **Preservar la Arquitectura de Mean Pooling Post-Hook:**
   Mantener la arquitectura actual donde el mean pooling se aplica **después** del hook (en `get_pooled_activations()`, líneas 214-217), NO dentro del hook. Esto preserva la posibilidad de análisis a nivel de token en el futuro.

3. **Limpieza Agresiva de Memoria (Método `clear_activations()`):**
   Actualizar el método en `mechanistic_hooks.py` (línea 220) para incluir garbage collection y liberación de caché GPU:
   ```python
   def clear_activations(self):
       self._activations.clear()
       import gc
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

4. **Naming Diferenciado de Archivos de Salida:**
   En `run_layer_probing.py` (línea 478), cuando se extraigan 40 capas, guardar con nombre diferente para no sobrescribir los resultados existentes:
   ```python
   num_layers = len(args.layers)
   npz_name = f"activations_{num_layers}layers.npz"
   save_path = os.path.join(args.output_dir, npz_name)
   ```
   De igual manera para el JSON de resultados (línea 497):
   ```python
   json_name = f"probing_results_{num_layers}layers.json"
   output_path = os.path.join(args.output_dir, json_name)
   ```

5. **Desacople y Serialización Estricta:**
   Asegurar que en `run_layer_probing.py` (línea 248) se mantenga `.numpy().copy()` al extraer los tensores. También añadir limpieza de caché GPU cada 50 videos (ya existe parcialmente en línea 256-258, solo reforzar con `torch.cuda.empty_cache()`).

6. **Logging WandB (mantener y parametrizar):**
   En `run_layer_probing.py`, el `--wandb_project` debe reflejar que es un run de 40 capas. Sugerir en el script Slurm:
   ```bash
   python -u run_layer_probing.py \
       --layers $(seq 0 39 | tr '\n' ' ') \
       --wandb_project videomae_probing \
       --output_dir output_dir
   ```
   Si WandB falla, el job NO debe abortar — la función `log_to_wandb()` ya está protegida por `if WANDB_AVAILABLE`.

7. **Script de Lanzamiento HPC (Slurm/Khipu) — `run_probing_40layers.sh`:**
   Crear el script Slurm completo para ejecutar en el clúster Khipu:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=probing_40layers
   #SBATCH --output=logs/probing_40layers_%j.out
   #SBATCH --error=logs/probing_40layers_%j.err
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --account=investigacion1
   #SBATCH --qos=a-investigacion1
   #SBATCH --time=2-00:00:00

   # 1. CARGA DE MÓDULOS BASE
   module load cuda/11.8
   module load miniconda/3.0

   # 2. ACTIVACIÓN DEL ENTORNO CONDA
   eval "$(conda shell.bash hook)"
   conda activate videomae_luis_izaguirre

   # 3. CONFIGURACIÓN DE RUTAS PARA TRITON (SOLUCIÓN stdlib.h)
   export CPATH=$CONDA_PREFIX/include:$CPATH
   export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

   # 4. PREVENCIÓN DE FRAGMENTACIÓN DE VRAM (CRÍTICO PARA 40 CAPAS)
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   # 5. AUTENTICACIÓN WANDB (usar key de ~/.netrc)
   export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')

   # 6. EJECUCIÓN DEL PIPELINE (PROBING)
   python -u run_layer_probing.py \
       --layers $(seq 0 39 | tr '\n' ' ') \
       --wandb_project videomae_probing \
       --output_dir output_dir

   # 7. GENERACIÓN DEL VIDEO DE CLUSTERING (PCA+UMAP)
   python -u visualize_clustering_evolution.py \
       --npz output_dir/activations_40layers.npz \
       --output_dir videos_simulation_clustering \
       --output_name clustering_evolution_ssv2_40layers.mp4
   ```

**NO modificar:**
- `models/modeling_finetune.py` (caja negra)
- `run_class_finetuning.py` (pipeline base inmutable)
- La lógica de `_make_hook()`, `_register_hooks()`, `get_activations()`, `get_pooled_activations()`
- La función `log_to_wandb()` existente (solo asegurarse de que sea invocada correctamente)

**Entregable:**
Proporciona los diffs de `mechanistic_hooks.py`, `run_layer_probing.py` y `visualize_clustering_evolution.py` (añadir argumento `--output_name` para naming personalizado del .mp4), más el script completo `run_probing_40layers.sh`.

**Requisitos de documentación del código:**
- Cada función nueva o modificada debe tener un **docstring** explicando su propósito, parámetros y valores de retorno.
- Cada decisión técnica (memoria, naming, limpieza de caché, etc.) debe estar **justificada con un comentario inline** explicando el *por qué* de esa implementación, no solo el *qué* hace.
- Los bloques de código críticos (hooks, limpieza de GPU, serialización de .npz) deben tener comentarios que expliquen la razón de diseño.
- Ejemplo de comentario esperado:
  ```python
  # Forzamos gc.collect() + empty_cache() después de cada video porque 
  # 40 capas × 3 componentes generan ~120 tensores de [1, 2048, 1408] 
  # que fragmentan la VRAM de la RTX A6000 si no se liberan activamente.
  ```

**Video de clustering existente que NO debe sobrescribirse:**
- `videos_simulation_clustering/clustering_evolution.mp4` (328 KB, 11 capas) — el nuevo video se llamará `clustering_evolution_ssv2_40layers.mp4`