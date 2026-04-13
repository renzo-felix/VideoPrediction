### Misión de Investigación: Arquitectura de Orquestación MLOps Desatendida para Clúster HPC (Slurm)

> **⏸️ ESTADO: En standby** — Depende de que el Paso 3 esté completado (hooks universales funcionando con V-JEPA2). Los campos marcados con `[⚠️ COMPLETAR]` deben ser rellenados antes de ejecutar. El `main.py` existente en la raíz del proyecto debe ser renombrado a `main_borrador.py` antes de ejecutar.

**Rol y Audiencia:**
Actúa como un Ingeniero MLOps Senior y Arquitecto de Software experto en despliegues en clústeres de Alto Rendimiento (HPC). El código que desarrolles orquestará el pipeline central de un equipo de investigadores que analiza cómo los modelos fundacionales (VideoMAEv2 y V-JEPA2) codifican propiedades físicas de los videos (velocidad) en sus dimensiones latentes.

**Contexto Científico y Operativo:**
Hemos desarrollado de forma modular los componentes del proyecto: extracción mecanicista no-destructiva (`mechanistic_hooks.py`), regresión logística lineal (`run_layer_probing.py`) y simulaciones topológicas UMAP (`visualize_clustering_evolution.py`).
Operamos en Khipu, un clúster HPC (GPU RTX A6000, 48GB VRAM). Dado que enviamos los trabajos a través de un gestor de colas (Slurm), la ejecución es **100% desatendida**. El uso de `input()` o interfaces interactivas está estrictamente prohibido.

**Estado Actual del Proyecto Verificado:**
- **Existe `VideoPrediction/main.py`** (64 líneas) que usa `subprocess` para llamar a scripts `.sh`. Renombrarlo a `main_borrador.py` antes de ejecutar este prompt.
- Los módulos internos están en `VideoMAEv2_Proyecto_Paper/`:
  - `mechanistic_hooks.py` → clase `UniversalVideoActivationExtractor`
  - `run_layer_probing.py` → funciones `extract_all_activations()`, `train_probes()`
  - `visualize_clustering_evolution.py` → clase `ClusteringEvolutionAnimator`
- El orquestador debe vivir en la **raíz del proyecto** (`VideoPrediction/`) porque es agnóstico al modelo.

**Checkpoints Verificados:**
```python
CHECKPOINT_MAP = {
    ("videomaev2", "ssv2"): "/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
    ("videomaev2", "k400"): "/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth",
    # [⚠️ COMPLETAR] Ruta al checkpoint de V-JEPA2 una vez descargado:
    ("vjepa2", "ssv2"): "/home/projects/video-prediction/checkpoints/vjepa2/vitg.pt",  # [⚠️ COMPLETAR]
    ("vjepa2", "k400"): "/home/projects/video-prediction/checkpoints/vjepa2/vitg.pt",  # [⚠️ COMPLETAR]
}
```
**¿Por qué completar?** El checkpoint de V-JEPA2 aún no ha sido descargado. La ruta y nombre del archivo dependen de la variante que elija el equipo.

**Datasets y CSVs Verificados:**
```python
CSV_MAP = {
    ("videomaev2", "ssv2"): "VideoMAEv2_Proyecto_Paper/physical_diagnostics.csv",
    ("videomaev2", "k400"): "VideoMAEv2_Proyecto_Paper/physical_diagnostics_k400.csv",
    # [⚠️ COMPLETAR] Rutas para V-JEPA2 (pueden ser los mismos CSVs si el proxy es idéntico):
    ("vjepa2", "ssv2"): "VideoMAEv2_Proyecto_Paper/physical_diagnostics.csv",  # [⚠️ COMPLETAR]
    ("vjepa2", "k400"): "VideoMAEv2_Proyecto_Paper/physical_diagnostics_k400.csv",  # [⚠️ COMPLETAR]
}
DATA_ROOT_MAP = {
    "ssv2": "VideoMAEv2_Proyecto_Paper/dataset/ssv2_luis",  # Frames .jpg
    "k400": "VideoMAEv2_Proyecto_Paper/k400",               # Videos .mp4
}
VIDEO_FORMAT_MAP = {
    "ssv2": "frames",   # load_video_frames() — lee .jpg/.png sueltos
    "k400": "mp4",      # load_video_mp4() — necesita decord o torchvision.io
}
NUM_CLASSES_MAP = {
    ("videomaev2", "ssv2"): 174,
    ("videomaev2", "k400"): 710,   # K710 checkpoint
    # [⚠️ COMPLETAR] V-JEPA2 no usa num_classes en el encoder (es un encoder puro):
    ("vjepa2", "ssv2"): None,  # [⚠️ COMPLETAR]
    ("vjepa2", "k400"): None,  # [⚠️ COMPLETAR]
}
```
**¿Por qué completar los de V-JEPA2?** V-JEPA2 es un encoder de representaciones (Joint-Embedding), NO un clasificador. No tiene `model.head` con `num_classes`. El modo de carga es diferente (ver Paso 3).

**Tu Tarea Exclusiva:**
Desarrollar `VideoPrediction/main_probing.py` y su script de despliegue `VideoPrediction/run_main.sh`.

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **Diseño de CLI Estricto (argparse):**
   ```python
   --model: choices=['videomaev2', 'vjepa2'], required=True
   --dataset: choices=['ssv2', 'k400'], required=True
   --run_probing: action='store_true'  # Extracción + regresión logística
   --run_visuals: action='store_true'  # PCA+UMAP + render MP4
   --layers: nargs='+', type=int, default=None  # None = todas las capas del modelo
   --max_videos: type=int, default=None  # Para debug rápido
   ```

2. **Pre-flight Checks (Validación Defensiva):**
   Antes de importar PyTorch, verificar:
   - Existencia del checkpoint (según `CHECKPOINT_MAP[(model, dataset)]`)
   - Existencia del dataset (según `DATA_ROOT_MAP[dataset]`)
   - Existencia del CSV de labels (según `CSV_MAP[(model, dataset)]`)
   - Si falta algo, `sys.exit(1)` con mensaje claro. No desperdiciar horas de cómputo.

3. **Manejo de Video Loaders:**
   El orquestador debe pasar el `video_format` correcto a `run_layer_probing.py`:
   - SSv2: `--video_format frames` (frames .jpg sueltos)
   - K400: `--video_format mp4` (archivos .mp4, requiere `decord`)

4. **Naming de Archivos de Salida:**
   Generar nombres diferenciados para no sobrescribir resultados entre modelos/datasets:
   ```python
   output_subdir = f"output_dir/{model}_{dataset}"
   npz_name = f"activations_{model}_{dataset}_{num_layers}layers.npz"
   json_name = f"probing_results_{model}_{dataset}.json"
   ```

5. **Gestión de Memoria entre Fases:**
   Si `--run_probing` y `--run_visuals`, forzar limpieza total entre fases:
   ```python
   # Fase 1: probing completo
   ...
   del model; gc.collect(); torch.cuda.empty_cache()
   # Fase 2: visualización (no necesita GPU)
   ...
   ```

6. **Script de Lanzamiento `run_main.sh`:**
   ```bash
   #!/bin/bash
   #SBATCH --partition=data-science
   #SBATCH --gres=gpu:1
   #SBATCH --job-name=probing_pipeline
   #SBATCH --output=logs/%x_%j.out
   #SBATCH --error=logs/%x_%j.err
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --account=investigacion1
   #SBATCH --qos=a-investigacion1
   #SBATCH --time=4-00:00:00
   
   module load cuda/11.8
   module load miniconda/3.0
   eval "$(conda shell.bash hook)"
   conda activate videomae_luis_izaguirre
   
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export CPATH=$CONDA_PREFIX/include:$CPATH
   export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
   
   # Ejemplo: VideoMAEv2 + K400 + Probing + Visuals
   python -u main_probing.py --model videomaev2 --dataset k400 --run_probing --run_visuals
   
   # [⚠️ COMPLETAR] Ejemplo para V-JEPA2 (descomentar cuando esté listo):
   # python -u main_probing.py --model vjepa2 --dataset ssv2 --run_probing --run_visuals
   ```

**Entregable Esperado:**
`main_probing.py` (con tipado estricto y lógica defensiva, `[⚠️ COMPLETAR]` marcados) y `run_main.sh`. Omitir introducciones redundantes.