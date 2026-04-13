> ### **Misión de Investigación: Abstracción Arquitectónica e Inyección Dinámica de Hooks en V-JEPA v2 y VideoMAEv2**
> 
> **Rol y Audiencia:**
> Actúa como un Arquitecto de Deep Learning Senior y experto en Interpretabilidad Mecanicista de Transformers. El código que generes será utilizado por un equipo de investigadores que evalúa si los modelos fundacionales de video aprenden propiedades físicas del mundo real (velocidad) de manera universal, ejecutando los experimentos en el clúster HPC Khipu (nodos con RTX A6000 de 48GB VRAM).
> 
> **Contexto Científico y Operativo:**
> Hemos comprobado que VideoMAEv2 (arquitectura Masked Autoencoder) decodifica la "velocidad" de forma robusta en su MLP final (96.5% accuracy en la Capa 39) al procesar el dataset SSv2. Nuestra meta es expandir el pipeline de manera agnóstica para evaluar **V-JEPA v2** y probar si esta cristalización semántica se replica en arquitecturas *Joint-Embedding* evaluando en paralelo sobre Kinetics-400 (K400). 
> Nuestro script de extracción actual (`mechanistic_hooks.py`) está hardcodeado para la topología y dimensiones del ViT-Giant de VideoMAEv2.
> 
> **Tu Tarea Exclusiva:**
> Refactoriza la clase extractora (creando una versión evolucionada, ej. `UniversalVideoActivationExtractor`) capaz de inyectar *Forward Hooks* no-destructivos de manera dinámica, adaptándose automáticamente a la topología del modelo que reciba por parámetro (VideoMAEv2 o V-JEPA v2).
> 
> **Especificaciones y Pasos de Implementación Obligatorios:**
> 
> 1. **Resolución Topológica Dinámica (Pathing Agnóstico):**
>    VideoMAEv2 almacena sus bloques en `model.blocks[i]`, mientras que V-JEPA utiliza una nomenclatura interna distinta (comúnmente `.transformer.blocks[i]` o similar). Desarrolla lógica usando *reflection* (`getattr` o recursividad) para identificar dinámicamente la ruta de la lista de bloques de atención y registrar los hooks iterativamente en sus 3 submódulos equivalentes: salida general (*Residual Stream*), salida de la atención (*MHA*) y salida del *MLP*.
> 
> 2. **Invarianza Dimensional Automática:**
>    En VideoMAEv2 la dimensión del embedding era estática (`1408`). V-JEPA puede diferir en tamaño. El extractor debe prescindir de números *hardcodeados* y capturar las dimensiones dinámicamente inspeccionando el tensor saliente (`tensor.shape[-1]`) en tiempo de ejecución.
> 
> 3. **Preservación Estricta de Gestión de Memoria (Anti-OOM):**
>    Es imperativo que la nueva clase abstracta mantenga las defensas arquitectónicas contra la fragmentación de VRAM en Khipu. 
>    * Dentro de cada *hook*, debes aplicar *Mean Pooling* espacial/temporal (`.mean(dim=1)`) **inmediatamente** para reducir el volumen de datos.
>    * Al transferir, exige explícitamente `.detach().cpu().numpy().copy()` para evitar *memory leaks* del grafo computacional.
>    * Mantén las rutinas de `torch.cuda.empty_cache()` en el ciclo de limpieza.
> 
> 4. **Comentarios de Conciliación de Grafos:**
>    El código debe estar exhaustivamente documentado. Añade un bloque de comentarios a nivel de clase explicando técnicamente cómo concilia tu código los grafos computacionales entre la familia MAE y la familia JEPA.
> 
> **Entregable Esperado:**
> Proporciona únicamente el código Python completo de la clase extractora refactorizada (`mechanistic_hooks.py` actualizado), tipado de forma estricta (Type Hints) y listo para producción. Omite tutoriales introductorios. Toda decisión de *pathing* y memoria debe estar rigurosamente justificada en los docstrings.