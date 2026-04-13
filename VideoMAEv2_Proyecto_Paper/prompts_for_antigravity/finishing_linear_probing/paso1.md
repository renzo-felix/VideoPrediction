> ### **Misión de Investigación: Refactorización Defensiva de Memoria para Extracción Exhaustiva Capa a Capa (VideoMAEv2)**
> 
> **Rol y Audiencia:**
> Actúa como un Ingeniero Senior de Machine Learning y MLOps especializado en Interpretabilidad Mecanicista, optimización de hardware a bajo nivel (PyTorch) y despliegue en entornos HPC (Slurm). Tu código será revisado por un equipo de investigadores analizando propiedades físicas (velocidad) en modelos fundacionales de video.
> 
> **Contexto y Problema Crítico:**
> Actualmente, nuestro script `mechanistic_hooks.py` extrae representaciones de 11 capas discretas de un modelo ViT-Giant (`embed_dim=1408`). Para lograr una animación de clustering (PCA+UMAP) que muestre una evolución semántica fluida e ininterrumpida, debemos escalar la extracción a las **40 capas completas**. 
> Esto generará 120 combinaciones de tensores por video (40 capas × 3 submódulos: Residual, MHA, MLP). Operamos en el clúster Khipu utilizando nodos con GPU NVIDIA RTX A6000 (48GB VRAM) bajo la partición `data-science`. Escalar a 40 capas sin una estrategia de memoria agresiva provocará un error de *Out-Of-Memory* (OOM) y Slurm asesinará el proceso.
> 
> **Tu Tarea:**
> Refactoriza y optimiza el script `mechanistic_hooks.py` actualizando la clase `VideoMAEActivationExtractor` para soportar la extracción de las 40 capas, aplicando estrictas buenas prácticas de HPC. No debes modificar los archivos del modelo original (arquitectura caja negra).
> 
> **Especificaciones y Pasos de Implementación Obligatorios:**
> 
> 1. **Inyección de Forward Hooks (Agnóstica):**
>    Itera sobre el rango completo de bloques (`range(40)`) y registra *hooks* en tres puntos críticos: salida del bloque (`residual`), salida de la atención (`attn` / `CosAttention`) y salida del perceptrón (`mlp`). 
> 
> 2. **Reducción de Dimensionalidad *In Situ* (Prevención de OOM):**
>    La salida cruda de ViT-Giant tiene la forma `[B, N, 1408]` (donde N ~ 2048 tokens espaciotemporales). Es **crítico** que el hook aplique *Mean Pooling* (`.mean(dim=1)`) inmediatamente dentro del paso de captura para reducir el tensor a `[B, 1408]` global por video. Extraer los 2048 tokens completos a la RAM para 40 capas destruirá la memoria.
> 
> 3. **Desacople y Serialización (Gestión RAM):**
>    Asegura el uso estricto de `.detach().cpu().numpy().copy()` al extraer los tensores reducidos. Esto rompe el grafo computacional y evita *memory leaks*. El resultado debe empaquetarse dinámicamente en un diccionario y guardarse en formato comprimido (`activations.npz`).
> 
> 4. **Limpieza Estricta de Caché GPU/CPU:**
>    Implementa un método de limpieza profunda tras el procesamiento de cada *batch*/video. Debes forzar la liberación de memoria usando `torch.cuda.empty_cache()` y recolección de basura con `gc.collect()` iterativamente.
> 
> 5. **Directivas de Sistema (Docstring):**
>    Documenta al inicio del script que la ejecución en bash requiere obligatoriamente exportar `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para evitar la fragmentación de la VRAM, lo cual es vital para tensores de 1408 dimensiones.
> 
> **Entregable:**
> Proporciona únicamente el código Python modular, crítico y altamente comentado de `mechanistic_hooks.py`. Omite explicaciones o tutoriales introductorios. Cada decisión relacionada al manejo de tensores y memoria debe estar justificada en los comentarios del código para validación científica.