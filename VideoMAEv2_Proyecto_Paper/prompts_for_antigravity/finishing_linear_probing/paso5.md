> ### **Misión de Investigación: Reporte Técnico Desatendido y Formulación Matemática de Vector Steering**
> 
> **Rol y Audiencia:**
> Actúa como el Investigador Principal (PI) de Deep Learning y experto en Interpretabilidad Mecanicista. El código que vas a generar servirá como la etapa final de un pipeline MLOps para un equipo de investigación que busca publicar un *paper* sobre cómo los modelos fundacionales (VideoMAEv2, V-JEPA) codifican la "velocidad" física en sus dimensiones latentes.
> 
> **Contexto Científico y Operativo:**
> A lo largo de nuestro pipeline, hemos validado que la propiedad semántica de "velocidad" cristaliza en la profundidad de la red, alcanzando una precisión de clasificación del 96.5% en la Capa 39, con un claro dominio del submódulo MLP sobre el de Atención. 
> Actualmente, tenemos cientos de logs `.json` en `output_dir/` provenientes de nuestros *Linear Probes*, y un archivo masivo comprimido `activations.npz` (con características de 1408 dimensiones por cada uno de los 21,202 videos).
> 
> **Tu Tarea Exclusiva:**
> Desarrolla un script de post-procesamiento unificado llamado `generate_technical_report.py`. Este script tendrá dos responsabilidades analíticas distintas pero complementarias, optimizadas para no exceder la RAM al cargar grandes datasets en un solo nodo.
> 
> **Especificaciones y Pasos de Implementación Obligatorios:**
> 
> 1. **Minería de Resultados (Agregador Automático):**
>    Programa un módulo que lea recursivamente los archivos `.json` de `output_dir/`. Debe parsear la estructura y exportar un `probing_summary_report.csv` consolidado. El CSV debe tener la estructura estricta: `Modelo, Dataset, Capa, Componente (Residual/MHA/MLP), Accuracy`. Esto alimentará nuestras tablas del paper.
> 
> 2. **Formulación Matemática del Vector Steering (Solo MLP Capa 39):**
>    A diferencia del Causal Patching de un solo par de videos, el *Steering* busca un sesgo direccional global. Escribe una función que cargue **únicamente** la matriz correspondiente al MLP de la Capa 39 desde `activations.npz` (para evitar *Out-Of-Memory*). 
>    * Cruza estos datos con las etiquetas de `physical_diagnostics.csv`.
>    * Calcula el centroide de los videos rápidos ($\mu_{fast}$) y el centroide de los lentos ($\mu_{slow}$).
>    * Computa el vector de dirección estandarizado: $V_{steer} = \frac{\mu_{fast} - \mu_{slow}}{||\mu_{fast} - \mu_{slow}||}$.
>    * Guarda este vector matemáticamente aislado en `steering_vector_c39_mlp.npy`.
> 
> 3. **Justificación Teórica en Formato Paper (Docstring):**
>    En la cabecera de la clase/función del Steering, escribe un docstring riguroso de nivel académico. Debes explicar matemáticamente cómo se utilizará este vector a futuro para sesgar el *forward pass* de la red neuronal mediante la suma de activaciones: $h' = h + \alpha V_{steer}$, donde $\alpha$ es el coeficiente de intervención. Relaciona esto con nuestro hallazgo de que el MLP actúa como un almacenamiento asimétrico *Key-Value* de semántica.
> 
> **Entregable Esperado:**
> El código fuente completo de `generate_technical_report.py`, tipado, modular y altamente eficiente en memoria utilizando iteradores o cargas condicionales de Numpy. Omitir introducciones genéricas; el tono debe ser estrictamente académico y arquitectónico.