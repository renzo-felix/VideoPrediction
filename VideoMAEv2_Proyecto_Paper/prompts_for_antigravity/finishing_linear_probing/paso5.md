### Misión de Investigación: Reporte Técnico Desatendido y (Opcional) Formulación Matemática de Vector Steering

> **✅ ESTADO: Parcialmente ejecutable** — Se puede ejecutar inmediatamente con los datos existentes de VideoMAEv2 + SSv2. La ejecución completa (todos los combos modelo×dataset) requiere que los pasos 1-4 estén finalizados.

**Rol y Audiencia:**
Actúa como el Investigador Principal (PI) de Deep Learning y experto en Interpretabilidad Mecanicista. El código que generes servirá como la etapa final del pipeline MLOps. Es un script de post-procesamiento para un equipo de investigación que busca presentar resultados sobre cómo los modelos fundacionales (VideoMAEv2, V-JEPA2) codifican la "velocidad" física en sus dimensiones latentes.

**Contexto Científico y Operativo:**
Hemos validado que la propiedad semántica de "velocidad" cristaliza en la profundidad de la red con VideoMAEv2 sobre SSv2 (96.5% accuracy en la Capa 39, dominio del MLP sobre Attention).

**Datos de Entrada Verificados (Estado Actual):**

| Archivo | Contenido | Combo |
|:---|:---|:---|
| `output_dir/probing_results.json` (7.5 KB) | 33 entradas `"block_{i}_{comp}"` → accuracy, confusion_matrix | VideoMAEv2 + SSv2 |
| `output_dir/activations.npz` (3.6 GB) | Claves: `block_{i}_{comp}_features` + `labels`. 21,202 videos × 11 capas × 3 comps | VideoMAEv2 + SSv2 |
| `physical_diagnostics.csv` (1.5 MB) | 21,202 videos, 5 columnas | SSv2 |

**Datos de Entrada Futuros (cuando se completen pasos 1-4):**

| Archivo esperado | Combo |
|:---|:---|
| `output_dir/probing_results_40layers.json` | VideoMAEv2 + SSv2 (40 capas) |
| `output_dir/probing_results_k400.json` | VideoMAEv2 + K400 |
| `output_dir/activations_40layers.npz` | VideoMAEv2 + SSv2 (40 capas) |
| `output_dir/activations_k400_Nlayers.npz` | VideoMAEv2 + K400 |
| `[⚠️ COMPLETAR]` | V-JEPA2 + SSv2 |
| `[⚠️ COMPLETAR]` | V-JEPA2 + K400 |

**¿Por qué marcar como completar?** Los nombres de archivo para V-JEPA2 dependen de cómo se configure el paso 4 (naming de salidas). El patrón será algo como `probing_results_vjepa2_ssv2.json`.

**Tu Tarea Exclusiva:**
Desarrollar `generate_technical_report.py` como script de post-procesamiento unificado con dos responsabilidades analíticas.

**Especificaciones y Pasos de Implementación Obligatorios:**

1. **CLI con argparse:**
   ```python
   --json_path: str, required  # Ruta al probing_results_*.json  
   --npz_path: str, default=None  # Ruta al activations_*.npz (solo para steering)
   --csv_path: str, default=None  # Ruta al physical_diagnostics*.csv (solo para steering)
   --model_name: str, required  # Para la columna "Modelo" del reporte (ej. "VideoMAEv2")
   --dataset_name: str, required  # Para la columna "Dataset" del reporte (ej. "SSv2")
   --output_dir: str, default="output_dir"
   --compute_steering: action='store_true'  # Opcional, calcular vector steering
   ```

2. **Minería de Resultados (Agregador Automático):**
   - Leer **un archivo** JSON a la vez (NO buscar "cientos" — cada ejecución genera un JSON)
   - Parsear claves `"block_{i}_{comp}"` para extraer `layer_idx` y `component`
   - Exportar `probing_summary_report.csv` con la estructura estricta:
     ```
     Modelo, Dataset, Capa, Componente, Accuracy
     ```
   - El modelo y dataset se reciben de `--model_name` y `--dataset_name`
   - **Para generar el reporte consolidado de todos los combos**, ejecutar el script múltiples veces y concatenar los CSVs:
     ```bash
     # VideoMAEv2 + SSv2
     python generate_technical_report.py --json_path output_dir/probing_results.json \
         --model_name VideoMAEv2 --dataset_name SSv2
     # VideoMAEv2 + K400
     python generate_technical_report.py --json_path output_dir/probing_results_k400.json \
         --model_name VideoMAEv2 --dataset_name K400
     # [⚠️ COMPLETAR] V-JEPA2 + SSv2
     # [⚠️ COMPLETAR] V-JEPA2 + K400
     ```

3. **(OPCIONAL) Formulación Matemática del Vector Steering — Solo MLP Capa 39:**
   Solo si `--compute_steering` es pasado y `--npz_path` y `--csv_path` son válidos.
   
   a. Cargar **únicamente** la matriz del MLP de la capa 39 desde el `.npz` para evitar OOM:
   ```python
   with np.load(npz_path) as data:
       features = data["block_39_mlp_features"]  # [N, embed_dim]
       labels = data["labels"]                     # [N]
   ```
   
   b. Cruzar con las etiquetas de velocidad (fast=1, slow=0).
   
   c. Calcular centroides:
   - μ_fast = mean de features donde labels==1
   - μ_slow = mean de features donde labels==0
   
   d. Computar vector de dirección estandarizado:
   V_steer = (μ_fast - μ_slow) / ||μ_fast - μ_slow||
   
   e. Guardar: `np.save("steering_vector_c39_mlp.npy", V_steer)`

4. **Justificación Teórica (si se computa steering):**
   En el docstring de la función de steering, incluir:
   - Fórmula de intervención: h' = h + α * V_steer, donde α es el coeficiente
   - Relación con la hipótesis de que el MLP actúa como almacenamiento asimétrico Key-Value
   - Referencia al 96.5% de accuracy como evidencia empírica de que la capa 39 MLP es el punto óptimo

5. **Eficiencia de Memoria:**
   - Usar `np.load(npz_path)` con context manager para liberar RAM
   - NO cargar todas las capas si solo se necesita la capa 39
   - Para el reporte CSV, usar iteradores o generadores

**Entregable Esperado:**
El código fuente completo de `generate_technical_report.py`, tipado, modular y altamente eficiente en memoria. Vector steering es OPCIONAL (controlado por flag CLI). Los combos de V-JEPA2 están marcados con `[⚠️ COMPLETAR]` para cuando estén disponibles.