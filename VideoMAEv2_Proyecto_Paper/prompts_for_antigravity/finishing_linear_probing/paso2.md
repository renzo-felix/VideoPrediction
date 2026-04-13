> ### **Misión de Investigación: Generalización Semántica y Pipeline End-to-End en Kinetics-400**
> 
> **Rol y Audiencia:**
> Actúa como un Data Scientist Senior y Arquitecto MLOps especializado en validación cruzada y evaluación de representaciones latentes en modelos fundacionales de video. Tu entregable será analizado por un equipo de investigadores que estudian cómo arquitecturas como VideoMAEv2 y V-JEPA codifican propiedades físicas (específicamente la velocidad) en sus activaciones internas.
> 
> **Contexto Científico y Operativo:**
> Hemos comprobado exitosamente nuestra hipótesis de "Cristalización Semántica" usando un proxy de velocidad sobre el dataset SSv2 (obteniendo un 96.5% de separabilidad lineal en el MLP de la Capa 39). Para asegurar que esto no es un sobreajuste al dataset, escalaremos el pipeline al masivo Kinetics-400 (K400).
> Los datos crudos residen en un enlace simbólico en Khipu: `/home/luis.izaguirre/video_features_proy/VideoPrediction/VideoMAEv2_Proyecto_Paper/k400`.
> 
> **Tu Tarea Exclusiva:**
> Desarrollar el código de preparación de datos (`create_physical_subset_k400.py`), su orquestador de Slurm (`run_subset_k400.sh`), y un plan de integración descendente, garantizando que no se quiebre el pipeline actual de extracción y *probing*.
> 
> **Especificaciones y Pasos de Implementación Obligatorios:**
> 
> 1. **Mapeo Antagonista (Proxy Físico):**
>    Analiza la ontología de las 400 clases de Kinetics-400. Selecciona e implementa diccionarios en Python aislando pares de acciones que representen "Alta Velocidad/Fuerza" (ej. *sprinting*, *slapping*, *bobsledding*) asignándoles la etiqueta `1`, frente a "Baja Velocidad/Inercia" (ej. *knitting*, *meditating*, *sneaking*) con la etiqueta `0`. Justifica científicamente tu selección de clases en el docstring del script, diferenciando la velocidad translacional/física del mero ruido visual.
> 
> 2. **Estructura de Salida Estandarizada:**
>    El script debe leer el directorio K400 y exportar un archivo CSV llamado `physical_diagnostics_k400.csv`. Para garantizar la compatibilidad retroactiva con nuestros scripts de Probing, el CSV **DEBE** contener estrictamente estas tres columnas: `video_path`, `original_label`, `speed_label`.
> 
> 3. **Script de Lanzamiento (HPC Khipu):**
>    Escribe `run_subset_k400.sh` con el boilerplate exacto para que sobreviva a las políticas de nuestro clúster:
>    * Partición: `--partition=data-science`
>    * Recursos y Permisos: `--gres=gpu:1`, `--mem=32G`, `--qos=a-investigacion1`, `--account=investigacion1`
>    * Entorno: Cargar `cuda/11.8`, `miniconda/3.0`, y activar `videomae_luis_izaguirre`
> 
> 4. **Análisis de Impacto Descendente (Markdown):**
>    Al final de tu respuesta, provee un plan crítico de máximo 15 líneas en formato Markdown. Explica detalladamente qué variables, rutas (paths) o dimensiones (shape assert) deben modificarse en los archivos subsiguientes (`run_layer_probing.py` y `visualize_clustering_evolution.py`) para que procesen el CSV de K400 fluidamente sin reescribir su núcleo matemático.
> 
> **Entregable Esperado:**
> El código fuente de `create_physical_subset_k400.py` altamente documentado, el script `run_subset_k400.sh`, y el bloque Markdown de integración. Omitir introducciones genéricas; el tono debe ser estrictamente técnico y metodológico.