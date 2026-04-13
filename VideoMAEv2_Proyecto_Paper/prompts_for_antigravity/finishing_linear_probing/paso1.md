### Misión de Investigación: Refactorización Defensiva de Memoria para Extracción Exhaustiva Capa a Capa (VideoMAEv2)

**Rol y Audiencia:**
Actúa como un Ingeniero Senior de Machine Learning y MLOps especializado en Interpretabilidad Mecanicista, optimización de hardware a bajo nivel (PyTorch) y despliegue en entornos HPC (Slurm). Tu código será revisado por un equipo de investigadores analizando propiedades físicas (velocidad) en modelos fundacionales de video.

**Contexto y Estado Actual del Proyecto (Verificado):**
Nuestro script `mechanistic_hooks.py` (248 líneas) contiene la clase `VideoMAEActivationExtractor` que extrae representaciones de **11 capas discretas** (`DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]` en la línea 68) de un modelo ViT-Giant (`embed_dim=1408`, `depth=40`).

Para lograr una animación de clustering (PCA+UMAP) que muestre una evolución semántica fluida e ininterrumpida, debemos escalar la extracción a las **40 capas completas**. Esto generará 120 combinaciones de tensores por video (40 capas × 3 submódulos: Residual, MHA, MLP).

**⚠️ LECCIONES APRENDIDAS DE EJECUCIONES ANTERIORES (CRÍTICO):**

Los siguientes errores ocurrieron en ejecuciones previas y DEBEN evitarse:

1. **Job 29935 — OOM por GPU incorrecta (Tesla T4, 16 GB):**
   La partición `gpu` de Khipu tiene 3 nodos con GPUs diferentes:
   - `g001`: Tesla T4 (16 GB VRAM) — **INSUFICIENTE** para ViT-Giant + 40 hooks
   - `g002`: RTX A6000 (48 GB VRAM) — **REQUERIDA**
   - `ag001`: A100 ×2 (80 GB VRAM) — suficiente pero suele estar ocupada
   
   **SOLUCIÓN:** Especificar `--gres=gpu:rtxa6000:1` y `--nodelist=g002` en el script Slurm para forzar el nodo correcto. NUNCA usar `--gres=gpu:1` genérico.

2. **Job 29937 — OOM por acumulación de listas en RAM (64 GB):**
   La función `extract_all_activations()` original acumulaba las activaciones en listas de Python (`all_pooled[name].append(tensor)`). Con 21,202 videos × 120 listas, el overhead de objetos Python fragmentaba ~40-50 GB de RAM, superando los 64 GB solicitados.
   
   **SOLUCIÓN:** Pre-asignar arrays numpy contiguos `np.empty((total, embed_dim), dtype=np.float32)` y escribir directamente con `all_features[name][idx] = tensor.numpy()`. Esto reduce el consumo a ~14.3 GB exactos (sin fragmentación). Solicitar `--mem=128G` para cubrir modelo + datos + overhead.

3. **Job 29937 — `NameError: name 'sys' is not defined` en `visualize_clustering_evolution.py`:**
   El módulo `sys` solo se importaba dentro de un bloque `except ImportError` pero se usaba en `main()` con `sys.exit(1)`.
   
   **SOLUCIÓN:** Añadir `import sys` al top-level del archivo.

4. **Advertencias no fatales (esperadas y aceptables):**
   - `ConvergenceWarning: lbfgs failed to converge`: Normal en capas tempranas (0-8) donde las features son menos linealmente separables. No afecta los resultados significativamente.
   - `wandb: Network error (ConnectionError)`: WandB puede tener problemas de red en nodos de cómputo. El script debe manejar esto con `try/except` o usar `--no_wandb` como fallback.

**Referencia de tiempos reales:**
- Job 29344 (11 capas): 4h 21min en data-science
- Job 29943 (40 capas, exitoso): ~16h en g002 (RTX A6000)

**Archivos existentes que NO deben sobrescribirse:**
- `output_dir/activations.npz` (3.6 GB) — contiene las activaciones de 11 capas × 3 componentes × 21,202 videos
- `output_dir/probing_results.json` (7.5 KB) — resultados de linear probing de las 11 capas

**Logging a WandB:**
El script `run_layer_probing.py` ya tiene integración con Weights & Biases. Incluye:
- Función `log_to_wandb()` que sube accuracy por capa, confusion matrices y gráficas interactivas
- Argumentos CLI: `--wandb_project` (default: `videomae_probing`) y `--no_wandb` para desactivar
- La API key de WandB ya está configurada en `~/.netrc` del clúster
- **OBLIGATORIO en el script Slurm:** Exportar la API key con:
  ```bash
  export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
  ```
- La función `log_to_wandb()` debe estar envuelta en `try/except` para que un error de red de WandB NO aborte el job completo después de horas de cómputo.

**Tu Tarea:**
Refactoriza `mechanistic_hooks.py` y actualiza `run_layer_probing.py` para soportar la extracción de las 40 capas completas, aplicando estrictas buenas prácticas de HPC. No debes modificar los archivos del modelo original (`models/modeling_finetune.py` — arquitectura caja negra).

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **Escalado de la lista de capas:**
   En `mechanistic_hooks.py` línea 68, cambiar:
   ```python
   DEFAULT_LAYERS = list(range(40))  # Las 40 capas completas
   ```
   Y en `run_layer_probing.py`, actualizar el default del argumento `--layers`:
   ```python
   parser.add_argument("--layers", type=int, nargs="+",
                       default=list(range(40)),
                       help="Capas a analizar (default: todas las 40 capas)")
   ```

2. **Preservar la Arquitectura de Mean Pooling Post-Hook:**
   Mantener la arquitectura actual donde el mean pooling se aplica **después** del hook (en `get_pooled_activations()`), NO dentro del hook. Esto preserva la posibilidad de análisis a nivel de token en el futuro.

3. **Limpieza Agresiva de Memoria (Método `clear_activations()`):**
   Actualizar el método en `mechanistic_hooks.py` para incluir garbage collection y liberación de caché GPU:
   ```python
   def clear_activations(self):
       self._activations.clear()
       import gc
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

4. **Naming Diferenciado de Archivos de Salida:**
   Cuando se extraigan 40 capas, guardar con nombre diferente para no sobrescribir los resultados existentes:
   ```python
   num_layers = len(args.layers)
   npz_name = f"activations_{num_layers}layers.npz"
   save_path = os.path.join(args.output_dir, npz_name)
   ```
   De igual manera para el JSON de resultados:
   ```python
   json_name = f"probing_results_{num_layers}layers.json"
   output_path = os.path.join(args.output_dir, json_name)
   ```

5. **⚠️ CRÍTICO — Pre-asignación de Arrays Numpy (evitar OOM en RAM):**
   La función `extract_all_activations()` **NO debe usar listas con `.append()`** para acumular activaciones. En su lugar:
   - Hacer un forward pass de prueba con el primer video para descubrir los nombres de activaciones y embed_dim
   - Pre-asignar arrays contiguos: `np.empty((total, embed_dim), dtype=np.float32)`
   - Escribir directamente: `all_features[name][write_idx] = tensor.squeeze(0).numpy()`
   - Recortar al final si hubo errores: `all_features[name][:write_idx]`
   
   Presupuesto de memoria estimado:
   - Arrays pre-asignados: 120 × [21202, 1408] × 4 bytes = **~14.3 GB**
   - Modelo ViT-Giant en GPU: ~6 GB VRAM
   - Overhead Python + imágenes: ~3-4 GB
   - **Total RAM necesaria: ~20 GB** (pedir 128 GB para margen de seguridad)

6. **Monitoreo de Memoria en Runtime:**
   Añadir monitoreo con `psutil` cada 500 videos para detectar memory leaks temprano:
   ```python
   if (i + 1) % 500 == 0:
       import psutil
       mem = psutil.Process().memory_info()
       print(f"  [MEM] Video {i+1}: RSS={mem.rss / 1e9:.1f} GB, VMS={mem.vms / 1e9:.1f} GB")
   ```

7. **Limpieza de caché GPU cada 50 videos:**
   ```python
   if (i + 1) % 50 == 0:
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

8. **Logging WandB (mantener y proteger contra errores de red):**
   La invocación de `log_to_wandb()` debe estar envuelta en `try/except`:
   ```python
   if not args.no_wandb and WANDB_AVAILABLE:
       try:
           log_to_wandb(results, args.wandb_project)
       except Exception as e:
           print(f"[ADVERTENCIA] WandB falló: {e}. Resultados ya guardados en JSON.")
   ```

9. **Fix de `import sys` en `visualize_clustering_evolution.py`:**
   Asegurar que `import sys` esté al top-level del archivo, no solo dentro de un `except ImportError`.

10. **Auto-detección de capas en `visualize_clustering_evolution.py`:**
    En lugar de hardcodear `self.layers = [0, 4, 8, ...]`, inspeccionar las claves del .npz para detectar automáticamente qué capas están disponibles. Esto permite que el mismo script funcione con .npz de 11 y 40 capas.

11. **Argumento `--output_name` en `visualize_clustering_evolution.py`:**
    Añadir argumento para naming personalizado del .mp4:
    ```python
    parser.add_argument("--output_name", type=str, default="clustering_evolution.mp4",
                        help="Nombre del archivo MP4 de salida")
    ```

12. **Script de Lanzamiento HPC (Slurm/Khipu) — `run_probing_40layers.sh`:**
    Crear el script Slurm completo para ejecutar en el clúster Khipu:
    ```bash
    #!/bin/bash
    #SBATCH --job-name=probing_40layers
    #SBATCH --output=logs/probing_40layers_%j.out
    #SBATCH --error=logs/probing_40layers_%j.err
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:rtxa6000:1
    #SBATCH --nodelist=g002
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=128G
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
- La función `log_to_wandb()` existente (solo asegurarse de que sea invocada correctamente y protegida con try/except)

**Entregable:**
Proporciona los diffs de `mechanistic_hooks.py`, `run_layer_probing.py` y `visualize_clustering_evolution.py` (añadir argumento `--output_name`, `import sys`, auto-detección de capas), más el script completo `run_probing_40layers.sh`.

**Requisitos de documentación del código:**
- Cada función nueva o modificada debe tener un **docstring** explicando su propósito, parámetros y valores de retorno.
- Cada decisión técnica (memoria, naming, limpieza de caché, pre-asignación de arrays, etc.) debe estar **justificada con un comentario inline** explicando el *por qué* de esa implementación, no solo el *qué* hace.
- Los bloques de código críticos (hooks, limpieza de GPU, serialización de .npz, pre-asignación) deben tener comentarios que expliquen la razón de diseño.
- Ejemplo de comentario esperado:
  ```python
  # Pre-asignamos arrays numpy contiguos en vez de listas con .append()
  # porque 21,202 videos × 120 listas fragmentan ~40-50 GB de RAM en
  # Python (overhead de objetos). Con pre-asignación, el consumo baja
  # a 14.3 GB exactos y el acceso es O(1) por índice.
  ```

**Video de clustering existente que NO debe sobrescribirse:**
- `videos_simulation_clustering/clustering_evolution.mp4` (328 KB, 11 capas) — el nuevo video se llamará `clustering_evolution_ssv2_40layers.mp4`