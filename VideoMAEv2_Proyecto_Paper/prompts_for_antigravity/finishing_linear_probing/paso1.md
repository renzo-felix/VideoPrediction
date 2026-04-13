### Misión de Investigación: Refactorización Defensiva de Memoria para Extracción Exhaustiva Capa a Capa (VideoMAEv2)

**Rol y Audiencia:**
Actúa como un Ingeniero Senior de Machine Learning y MLOps especializado en Interpretabilidad Mecanicista, optimización de hardware a bajo nivel (PyTorch) y despliegue en entornos HPC (Slurm). Tu código será revisado por un equipo de investigadores analizando propiedades físicas (velocidad) en modelos fundacionales de video.

**Contexto y Estado Actual del Proyecto (Verificado):**
Nuestro script `mechanistic_hooks.py` (248 líneas) contiene la clase `VideoMAEActivationExtractor` que extrae representaciones de **11 capas discretas** (`DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]` en la línea 68) de un modelo ViT-Giant (`embed_dim=1408`, `depth=40`).

Para lograr una animación de clustering (PCA+UMAP) que muestre una evolución semántica fluida e ininterrumpida, debemos escalar la extracción a las **40 capas completas**. Esto generará 120 combinaciones de tensores por video (40 capas × 3 submódulos: Residual, MHA, MLP).

Operamos en el clúster Khipu utilizando nodos con GPU NVIDIA RTX A6000 (48GB VRAM) bajo la partición `data-science`.

**Archivos existentes que NO deben sobrescribirse:**
- `output_dir/activations.npz` (3.6 GB) — contiene las activaciones de 11 capas × 3 componentes × 21,202 videos
- `output_dir/probing_results.json` (7.5 KB) — resultados de linear probing de las 11 capas

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

6. **Directivas de Sistema (Docstring):**
   Añadir en la cabecera de `mechanistic_hooks.py`:
   ```python
   # REQUISITO DE EJECUCIÓN HPC (bash):
   # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   # Esto evita la fragmentación de VRAM con tensores de 1408 dimensiones.
   ```

**NO modificar:**
- `models/modeling_finetune.py` (caja negra)
- `run_class_finetuning.py` (pipeline base inmutable)
- La lógica de `_make_hook()`, `_register_hooks()`, `get_activations()`, `get_pooled_activations()`

**Entregable:**
Proporciona únicamente los diffs de `mechanistic_hooks.py` y `run_layer_probing.py`. Cada decisión de memoria debe estar justificada en los comentarios del código.