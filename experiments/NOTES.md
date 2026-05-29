# Notas: Experimentos de Predicción de Video y Vectores de Dirección

Estas notas resumen el funcionamiento de los scripts y módulos de la carpeta de experimentos, orientadas a la presentación al grupo de investigación. El objetivo principal de estos experimentos es extraer "vectores de dirección" (steering vectors) que representen conceptos puros como la **velocidad** dentro del espacio latente de modelos fundacionales de video (V-JEPA y VideoMAE), y comprobar su capacidad de generalización sobre objetos no vistos durante el entrenamiento.

## Archivos Principales

### 1. `config.py`
Este archivo centraliza todas las variables globales y constantes de configuración del proyecto:
- **Directorios y rutas:** Ubicación de modelos (`models/`), datos generados (`videos/`), extracciones (`activations/`) y resultados.
- **Configuraciones de los Modelos:** Capas específicas y dimensionalidad oculta para V-JEPA y VideoMAE (por defecto capa 9 y dimensión 1408).
- **Parámetros de Simulación:** Define la cámara, tamaños, colores, formas y texturas disponibles para los objetos en PyBullet.
- **Hiperparámetros de Experimentos:** Tamaño del batch, rango de velocidades a simular y configuraciones para los métodos estadísticos (ej. percentiles).

### 2. `src/generate_videos.py`
Script fundamental que utiliza la biblioteca de simulación 3D **PyBullet** para generar los videos que servirán de dataset. Posee dos modos de funcionamiento:
- **Modo de Entrenamiento (`--mode training`):** Crea un conjunto de videos base controlados. Genera esferas de diferentes tamaños (pequeño, mediano, grande) y colores, desplazándose a distintas velocidades. Esto asegura que el modelo tenga representaciones suficientes para separar la velocidad del tamaño o el color.
- **Modo de Prueba (`--mode test`):** Genera videos con **objetos composicionalmente nuevos** (formas, colores o texturas que no se vieron juntas durante el entrenamiento, o que son completamente nuevas, como formas cúbicas o texturas ajedrezadas). Esto permite evaluar si el concepto de "velocidad" aprendido por el vector generaliza más allá del objeto en sí.

### 3. `src/extract_activations.py`
Se encarga de procesar los videos generados a través de los modelos de video (V-JEPA o VideoMAE).
- **Proceso:** Carga los videos en lotes y los pasa por el modelo seleccionado. A través de un *hook* en PyTorch (`ActivationExtractor`), intercepta y extrae la salida de una capa intermedia oculta del modelo (por defecto, la capa 9).
- **Resultado:** Guarda matrices de Numpy (`.npy`) que contienen las activaciones para cada video, vinculándolas con la meta-data de velocidad real obtenida de la simulación. 

### 4. `src/train_steering.py`
Es el núcleo analítico para aislar el concepto de velocidad dentro de las activaciones.
- **Objetivo:** Calcular un *steering vector* que represente la "velocidad", calculando la diferencia entre las activaciones de objetos rápidos y objetos lentos.
- **Métodos Implementados:** 
  - *Percentil:* Simplemente resta las activaciones del percentil alto de velocidad frente al bajo.
  - *Estratificado (Recomendado):* Balancea distribuciones problemáticas. Por ejemplo, asegura que al restar grupos de alta y baja velocidad, haya la misma proporción de esferas grandes y pequeñas en ambos grupos para evitar que el vector final arrastre sesgos de "tamaño".
- **Evaluación Interna:** Entrena un predictor lineal y realiza validación cruzada. También lo compara contra *baselines* como PCA, para validar que la dirección encontrada sea robusta.

### 5. `src/test_transfer.py`
Mide la calidad y generalización del vector de dirección aprendido.
- **Proceso:** Toma el vector de dirección generado en el script de entrenamiento, lee las activaciones de los conjuntos de prueba (que incluyen colores, texturas, formas y materiales nuevos) y proyecta estas activaciones sobre el vector.
- **Métricas:** Evalúa las predicciones de velocidad utilizando coeficientes de correlación de Pearson y Spearman, así como MAE y RMSE.
- **Importancia:** Confirma si el modelo fundacional está desenredando la noción de movimiento físico independientemente de la apariencia visual del objeto (cero solapamiento entre tren/prueba).

## Estructura de Directorios Internos (`src/`)

Para mantener el código escalable y ordenado, el proyecto descompone la lógica en varios sub-módulos:
- **`src/data/`**: Contiene la lógica para invocar a PyBullet (`pybullet_objects.py`, `textures.py`) y los `Dataset` y `DataLoader` de PyTorch (`video_loaders.py`).
- **`src/models/`**: Scripts dedicados a cargar las arquitecturas de V-JEPA y VideoMAE y proporcionar utilidades de extracción limpia de activaciones.
- **`src/steering/`**: Funciones matemáticas puras donde residen los métodos de cálculo de diferencia de medias, evaluación de percentiles y comparativas (PCA, regresión Ridge).
- **`src/analysis/`**: Utilidades para leer, transformar y alinear rápidamente los metadatos y las activaciones guardadas en disco.

## Conclusión para la Presentación
El código está diseñado bajo un estricto rigor experimental:
1. **Control de Confusores:** Gracias a la simulación y al método estratificado, sabemos que si un modelo asocia "objeto grande" con "baja velocidad" de forma errónea, el análisis permite desenredarlo y analizar su impacto.
2. **Pruebas Out-of-Distribution (OOD):** Evaluar sobre categorías nunca antes vistas (ej. cilindros metálicos gigantes) demuestra la universalidad de la representación de movimiento adquirida, confirmando si el concepto visual es realmente independiente del objeto en movimiento.
