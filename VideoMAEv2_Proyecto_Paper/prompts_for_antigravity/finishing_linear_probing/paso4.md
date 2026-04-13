> ### **Misión de Investigación: Arquitectura de Orquestación MLOps Desatendida para Clúster HPC (Slurm)**
> 
> **Rol y Audiencia:**
> Actúa como un Ingeniero MLOps Senior y Arquitecto de Software experto en despliegues en clústeres de Alto Rendimiento (HPC). El código que desarrolles orquestará el pipeline central de un equipo de investigadores que analiza cómo los modelos fundacionales (VideoMAEv2 y V-JEPA) codifican propiedades físicas de los videos (velocidad) en sus dimensiones latentes.
> 
> **Contexto Científico y Operativo:**
> Hemos desarrollado de forma modular los componentes del proyecto: extracción mecanicista no-destructiva (`mechanistic_hooks.py`), regresión logística lineal (`run_layer_probing.py`) y simulaciones topológicas UMAP (`visualize_clustering_evolution.py`). 
> Operamos en Khipu, un clúster HPC (GPU RTX A6000, 48GB VRAM). Dado que enviamos los trabajos a través de un gestor de colas (Slurm), la ejecución es **100% desatendida**. El uso de `input()` o interfaces interactivas está estrictamente prohibido, ya que congelaría los nodos del servidor de forma permanente.
> 
> **Tu Tarea Exclusiva:**
> Desarrolla el orquestador maestro `main.py` utilizando la librería estándar `argparse` y su correspondiente script de despliegue `run_main.sh`. Este orquestador actuará como el único punto de entrada (Entry Point) al proyecto, unificando la extracción, el probing y la visualización.
> 
> **Especificaciones y Pasos de Implementación Obligatorios:**
> 
> 1. **Diseño de CLI Estricto (argparse):**
>    Implementa las siguientes banderas (*flags*) con validación de tipos y opciones restringidas (choices):
>    * `--model`: Opciones permitidas `['videomaev2', 'vjepa']`. Requerido.
>    * `--dataset`: Opciones permitidas `['ssv2', 'k400']`. Requerido.
>    * `--run_probing`: Flag booleano (`action='store_true'`). Ejecuta la fase de extracción de features y regresión logística.
>    * `--run_visuals`: Flag booleano (`action='store_true'`). Ejecuta la fase de reducción dimensional (PCA+UMAP) y renderizado MP4.
> 
> 2. **Pre-flight Checks (Validación Defensiva):**
>    Antes de iniciar cualquier importación pesada (PyTorch/Transformers) o asignar VRAM, el script debe verificar obligatoriamente la existencia de rutas críticas dependiendo de los argumentos:
>    * La existencia del enlace simbólico de checkpoints (`/home/projects/video-prediction/checkpoints`).
>    * La existencia del dataset solicitado (la carpeta `k400/` o el directorio local de SSv2).
>    Si algo falta, debe hacer un `sys.exit(1)` con un log claro para no desperdiciar horas de cómputo en Slurm.
> 
> 3. **Gestión de Memoria y Ruteo Global:**
>    El orquestador debe importar los módulos internos (Hooks, Probing, Visualize) e inyectarles los parámetros del CLI. **Crítico:** Si el usuario solicita tanto `--run_probing` como `--run_visuals`, el bloque condicional entre ambos debe forzar una recolección de basura total (`gc.collect()`) y vaciar la caché de la GPU (`torch.cuda.empty_cache()`) para evitar que el estado remanente del probing ahogue la generación visual.
> 
> 4. **Script de Lanzamiento (Boilerplate HPC Khipu):**
>    Escribe `run_main.sh` mostrando cómo invocar este CLI para un caso de uso completo (VideoMAEv2 + K400 + Probing + Visuals). Debes incluir **obligatoriamente** el siguiente bloque de recursos para Slurm:
>    * `--partition=data-science`, `--gres=gpu:1`, `--mem=32G`
>    * Las credenciales de investigación: `--qos=a-investigacion1` y `--account=investigacion1`.
>    * Las mitigaciones de fragmentación e inclusión C++: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `export CPATH=$CONDA_PREFIX/include:$CPATH` y `LIBRARY_PATH`.
> 
> **Entregable Esperado:**
> El código fuente de `main.py` (con tipado estricto y lógica defensiva) y el script bash `run_main.sh`. Omite introducciones redundantes; el tono debe ser directamente aplicable, estructurado y documentado a nivel de ingeniería de sistemas de aprendizaje automático.