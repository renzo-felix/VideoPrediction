### **Misión de Investigación: Evolución del Clustering Mecanicista (VideoMAEv2 y V-JEPA)**

**Rol y Contexto:**
Actúa como un Ingeniero e Investigador de Deep Learning Senior especializado en **Interpretabilidad Mecanicista**. Nuestra jefa de investigación nos ha dado las siguientes directrices estrictas:
1. El parcheo causal (Activation Patching) queda descartado por ahora. El foco 100% es el **Linear Probing** y el **Clustering de activaciones**.
2. El pipeline de visualización y probing **debe ser agnóstico del modelo** (debe funcionar igual para VideoMAEv2 o V-JEPA).
3. Haremos un *sanity check* para ver cómo evoluciona la separabilidad semántica del modelo evaluado en distintos datasets (`SSv2` y más adelante `IntPhys` con velocidades continuas y pelotas de distintos materiales).

**El Estado Actual del Proyecto (¡NO REESCRIBIR ESTO!):**
Ya hemos codificado en PyTorch un pipeline usando Forward Hooks (`mechanistic_hooks.py`) que extrae con éxito las activaciones (en el Residual Stream, Attention y MLP) de las capas `[0, 8, 16, 24, 32, 39]` para 21,202 videos. El modelo se ejecuta sin entrenar (`eval()`) y extrae todo en matrices NumPy/Tensores listos para usarse.

---

### **Tu Tarea Exclusiva: Fase Visual de Clustering Evolutivo**

Crea un único script de Python llamado `visualize_clustering_evolution.py` y su respectivo script de Slurm (`run_vis.sh`). Este código actuará sobre las activaciones ya extraídas y su objetivo es **generar una animación (.mp4 o .gif)** que muestre visualmente cómo la red neuronal va ordenando los conceptos capa por capa.

**Requisitos Computacionales del Script (`visualize_clustering_evolution.py`):**
1. **Reducción de Dimensionalidad:** Deberá recibir matrices de *features* de dimensiones altas (ej. las de ViT de `1408` dimensiones). Debes aplicar **PCA** rápido para reducir el ruido a 50 dimensiones, y luego **t-SNE** o **UMAP** para reducirlas a 2 dimensiones (2D).
2. **Alineación Inter-capa:** Asegúrate de que los ejes espaciales entre la reducción de la Capa 0 y la Capa 39 mantengan cierta coherencia de distancias semánticas; de lo contrario, la animación saltará caóticamente y será incomprensible visualmente. Justifica los hiperparámetros (ej: *random_state*, *init='pca'* en UMAP).
3. **Agrupación y Colaboración AGNÓSTICA:** El script no invoca al modelo de video en absoluto. Solo recibe un dataset `X` (lista de arrays de características de cada capa) y un dataset `Y` (etiquetas binarias Rápidas/Lentas, o variables contínuas 0 a 100 de Intphys coloreadas mediante un colormap progresivo).
4. **Animación (Matplotlib/ImageIO):** Deberá crear un scatter plot donde fotograma a fotograma evolucionemos: Capa 0 -> Capa 8 -> Capa 16... hasta la 39, generando un mp4 donde se vea cómo las pelotitas (los videos) se van separando por su propiedad de velocidad.

---

### **Protocolo de Ejecución en HPC (Khipu)**
Crea el archivo `run_vis.sh` asegurándote de usar estos parámetros críticos o será asesinado por los administradores:
* **Recursos SBATCH:** `--partition=data-science`, `--gres=gpu:1`, `--mem=32G`, `--time=4-00:00:00`, y **ESTRICTAMENTE AÑADIR** `--qos=a-investigacion1` y `--account=investigacion1`.
* **Entorno Bash:** `module load cuda/11.8`, `module load miniconda/3.0`, y activar el entorno `videomae_luis_izaguirre`.
* **Manejo de Librerías del Cluster:** Debes incluir las exportaciones `CPATH=$CONDA_PREFIX/include:$CPATH` y `LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH`.

No me escribas tutoriales básicos. Dame el código Python estructurado al nivel de investigación de DeepMind/OpenAI resuelto en clases separadas y el script de Slurm listo para ser despachado al servidor.