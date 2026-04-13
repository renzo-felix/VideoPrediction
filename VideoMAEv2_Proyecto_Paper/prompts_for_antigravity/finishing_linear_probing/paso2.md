### Misión de Investigación: Generalización Semántica y Pipeline End-to-End en Kinetics-400

**Rol y Audiencia:**
Actúa como un Data Scientist Senior y Arquitecto MLOps especializado en validación cruzada y evaluación de representaciones latentes en modelos fundacionales de video. Tu entregable será analizado por un equipo de investigadores que estudian cómo arquitecturas como VideoMAEv2 codifican propiedades físicas (específicamente la velocidad) en sus activaciones internas.

**Contexto Científico y Operativo:**
Hemos comprobado exitosamente nuestra hipótesis de "Cristalización Semántica" usando un proxy de velocidad sobre el dataset SSv2 (obteniendo un 96.5% de accuracy en la Capa 39). Para asegurar que esto no es un sobreajuste al dataset, escalaremos el pipeline al masivo Kinetics-400 (K400).

**Estructura de Datos Verificada en el Clúster:**
- **Videos K400 (.mp4):** Enlace simbólico en `VideoMAEv2_Proyecto_Paper/k400 -> /home/datasets/k400/val` (19,881 archivos .mp4)
- **Labels K400:** En `VideoMAEv2_Proyecto_Paper/dataset/k400_luis/labels/` con archivos:
  - `val.csv` — 19,779 entradas, formato: `k400/NOMBRE_VIDEO.mp4 CLASS_ID` (separado por espacio)
  - `train.csv` — 240,436 entradas, mismo formato
- **Mapa de clases:** `VideoMAEv2_Proyecto_Paper/label_map_k710.txt` — 709 líneas, donde el número de línea (base 0) es el class_id y el contenido es el nombre de la acción. K400 usa class_ids 0-399 (subconjunto de K710).
- **Checkpoint K400:** `/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth` (finetuned en K710 que incluye K400, 710 clases)
- **Formato de video:** K400 usa archivos .mp4 directo, **NO** frames sueltos como SSv2. Esto requiere un video loader diferente.

**Entorno HPC Khipu:**
- **Entorno conda:** `videomae_luis_izaguirre`
- **Módulos del sistema:** `cuda/11.8`, `miniconda/3.0`
- **Exports obligatorios:**
  ```bash
  export CPATH=$CONDA_PREFIX/include:$CPATH
  export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

**Logging a WandB:**
El script `run_layer_probing.py` ya tiene integración completa con Weights & Biases:
- Función `log_to_wandb()` que sube accuracy por capa, confusion matrices y gráficas interactivas
- Argumentos CLI: `--wandb_project` (default: `videomae_probing`) y `--no_wandb` para desactivar
- La API key de WandB ya está configurada en `~/.netrc` del clúster
- Para K400, usar un proyecto/run name diferente para distinguir de SSv2: `--wandb_project videomae_probing_k400`
- Para garantizar la autenticación en Slurm, exportar la key:
  ```bash
  export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
  ```

**Proxy de velocidad existente en SSv2 (referencia para replicar):**
En `create_physical_subset.py`, se usaron pares antagonistas semánticos como:
- Fast: "Throwing something", "Something falling like a rock" 
- Slow: "Holding something", "Poking something so lightly..."

**Tu Tarea Exclusiva:**
Desarrollar `create_physical_subset_k400.py`, su orquestador Slurm `run_subset_k400.sh`, y un plan de integración descendente.

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **Mapeo Antagonista (Proxy Físico) — Verificado contra `label_map_k710.txt`:**
   
   Antes de codificar los diccionarios, lee el archivo `label_map_k710.txt` para obtener las clases verificadas. A continuación los class_ids VERIFICADOS de K710 que mapean a velocidad. Revisa y justifica cada selección en el docstring:

   **Alta Velocidad / Alta Energía Cinética (speed_label=1):**
   | K710 ID | Nombre | Justificación |
   |:---:|:---|:---|
   | 11 | bobsledding | Velocidad translacional extrema |
   | 22 | pole vault | Aceleración + desplazamiento vertical rápido |
   | 28 | skateboarding | Velocidad translacional sostenida |
   | 29 | dunking basketball | Salto + impacto de alta energía |
   | 46 | ski jumping | Velocidad extrema + vuelo |
   | 54 | kicking field goal | Transferencia de energía cinética al balón |
   | 132 | slapping | Velocidad de impacto manual |
   | 155 | throwing discus | Lanzamiento atlético con momento angular |
   | 222 | shot put | Lanzamiento pesado con fuerza máxima |
   | 267 | hammer throw | Velocidad angular + lanzamiento |
   | 295 | javelin throw | Proyección de alta velocidad |
   | 299 | long jump | Sprint + salto horizontal |
   | 300 | parkour | Movimiento acrobático rápido y continuo |

   **Baja Velocidad / Baja Energía Cinética (speed_label=0):**
   | K710 ID | Nombre | Justificación |
   |:---:|:---|:---|
   | 37 | stretching leg | Movimiento lento, controlado |
   | 123 | folding clothes | Manipulación manual lenta |
   | 129 | tai chi | Movimiento deliberadamente lento |
   | 219 | ironing | Desplazamiento mínimo y repetitivo |
   | 248 | knitting | Casi estático, movimiento fino de dedos |
   | 249 | reading book | Estático, sin movimiento significativo |
   | 262 | stretching arm | Movimiento lento, controlado |
   | 290 | folding napkins | Manipulación manual mínima |
   | 311 | playing chess | Estático, movimiento mínimo |
   | 319 | arranging flowers | Manipulación cuidadosa y lenta |
   | 331 | watering plants | Movimiento suave, sin prisa |
   | 371 | yoga | Posturas estáticas o transiciones lentas |
   | 515 | calligraphy | Movimiento fino, controlado |

   **IMPORTANTE:** Estos class_ids son del label_map_k710.txt (0-indexed por línea). Los CSVs de K400 (`val.csv`, `train.csv`) usan class_ids 0-399. Debes leer `label_map_k710.txt` para mapear nombre↔id y luego cruzar con los class_ids del CSV. Si algún class_id de la tabla anterior es ≥400 (como 515=calligraphy), **no existirá** en el dataset K400 y debe ser excluido.

2. **Estructura de Salida Estandarizada (Compatibilidad con SSv2):**
   El CSV debe llamarse `physical_diagnostics_k400.csv` y tener las **mismas 5 columnas** que `physical_diagnostics.csv` de SSv2 para compatibilidad retroactiva con `run_layer_probing.py`:
   ```
   video_path, num_frames, original_label, class_id, speed_label
   ```
   Para obtener `num_frames`, puedes usar `decord` o `cv2.VideoCapture` para contar los frames del .mp4.

3. **Lectura de datos:**
   - Para los labels: leer `dataset/k400_luis/labels/val.csv` (formato: `k400/VIDEO.mp4 CLASS_ID`, separado por espacio)
   - Para el mapa de nombres: leer `label_map_k710.txt` donde el número de línea (base 0) = class_id
   - Para verificar que los videos existen: comprobar en el directorio `k400/`

4. **Script de Lanzamiento HPC (Slurm/Khipu) — `run_subset_k400.sh`:**
   Este paso incluye crear el CSV y ejecutar el probing completo en K400, por lo que necesita GPU:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=probing_k400
   #SBATCH --output=logs/probing_k400_%j.out
   #SBATCH --error=logs/probing_k400_%j.err
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

   # 4. PREVENCIÓN DE FRAGMENTACIÓN DE VRAM
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   # 5. AUTENTICACIÓN WANDB (usar key de ~/.netrc)
   export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')

   # 6. PASO A: Crear subset físico de K400
   python -u create_physical_subset_k400.py

   # 7. PASO B: Ejecutar probing en K400
   python -u run_layer_probing.py \
       --wandb_project videomae_probing_k400 \
       --output_dir output_dir

   # 8. PASO C: Generar video de clustering (PCA+UMAP) para K400
   python -u visualize_clustering_evolution.py \
       --npz output_dir/activations_k400_Nlayers.npz \
       --output_dir videos_simulation_clustering \
       --output_name clustering_evolution_k400.mp4
   ```

5. **Análisis de Impacto Descendente (Markdown, max 15 líneas):**
   Explica qué debe modificarse en los archivos subsiguientes:
   - `run_layer_probing.py`: Necesita un **nuevo video loader para .mp4** (la función `load_video_frames()` actual lee frames .jpg/.png sueltos, que es el formato SSv2). Para K400 se necesita `load_video_mp4()` usando `decord.VideoReader` o `torchvision.io.read_video()`. Parametrizar con `--video_format [frames|mp4]`.
   - `run_layer_probing.py`: El `num_classes=174` está hardcodeado para SSv2 en `load_model()` (línea 137). Para K400 con checkpoint K710, debe ser `num_classes=710`.
   - `visualize_clustering_evolution.py`: Añadir argumento `--output_name` para personalizar el nombre del .mp4. El video de K400 se guardará como `clustering_evolution_k400.mp4` en `videos_simulation_clustering/`.
   - Crear activaciones con nombre diferenciado: `activations_k400_Nlayers.npz` y `probing_results_k400.json`
   - Usar `--wandb_project videomae_probing_k400` para separar los resultados de K400 vs SSv2 en WandB

**Entregable Esperado:**
El código fuente de `create_physical_subset_k400.py` altamente documentado, el script `run_subset_k400.sh`, el diff de `visualize_clustering_evolution.py` (argumento `--output_name`), y el bloque Markdown de integración. Omitir introducciones genéricas; el tono debe ser estrictamente técnico y metodológico.

**Requisitos de documentación del código:**
- Cada función debe tener un **docstring** completo explicando su propósito, parámetros y valores de retorno.
- Los diccionarios de mapeo de clases (alta/baja velocidad) deben tener **comentarios inline** que justifiquen la clasificación de cada clase desde el punto de vista de la física (velocidad translacional, energía cinética, etc.).
- Cada decisión técnica debe estar **justificada con un comentario explicando el *por qué***, no solo el *qué* hace.
- Las funciones de lectura de CSV y cruce con `label_map_k710.txt` deben tener comentarios explicando la lógica de mapeo entre class_ids de K400 (0-399) y los K710 IDs.
- Ejemplo de comentario esperado:
  ```python
  # Excluimos class_ids >= 400 porque el dataset K400 solo contiene
  # las primeras 400 clases de K710. Clases como 'calligraphy' (id=515)
  # existen en K710 pero NO tienen videos en nuestro subset de K400.
  ```

**Videos de clustering existentes que NO debe sobrescribirse:**
- `videos_simulation_clustering/clustering_evolution.mp4` (328 KB, 11 capas SSv2) — el nuevo video de K400 se llamará `clustering_evolution_k400.mp4`