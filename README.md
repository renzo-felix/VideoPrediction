# 🎬 Interpretabilidad Mecanística en Modelos de Video

> **Investigación activa** — Universidad de Ingeniería y Tecnología (UTEC) · Dic. 2025 – actualidad

---

## 📌 Descripción del Proyecto

Este repositorio contiene el trabajo de investigación sobre **interpretabilidad mecanística** aplicada a modelos de comprensión de video. El objetivo central es realizar un **análisis de causalidad** sobre dos modelos de representación de video de alto rendimiento —**VideoMAEv2** y **V-JEPA 2**— con el fin de determinar si, en sus capas internas, han aprendido propiedades físicas del mundo real de forma implícita.

La motivación de fondo es adquirir la capacidad de **rastrear la ruta de aprendizaje** de estos modelos y tener control sobre qué aspectos de la realidad aprenden, sentando las bases para el diseño y entrenamiento de futuros **world models**.

### ¿Qué se investiga?

- ¿Aprenden los modelos de video conceptos físicos (gravedad, inercia, causalidad temporal) sin supervisión explícita?
- ¿Qué tipo de representaciones internas emergen en cada capa del Transformer?
- ¿Es posible identificar y auditar estas representaciones para tener control sobre el aprendizaje?

---

## 👥 Equipo

| Integrante | Rol | Modelo |
|---|---|---|
| **Luis Izaguirre** | Investigador | VideoMAEv2 |
| **Renzo Felix** | Investigador | V-JEPA 2 |
| **Ariana Villegas** | Asesora e Investigadora | VideoMAEv2 y V-JEPA 2 |

Proyecto realizado en el clúster de alta computación **Khipu** de UTEC, bajo la cuenta de investigación `investigacion1`.

---

## 🗂️ Estructura del Repositorio

```
VideoPrediction/
│
├── main.py                        # Lanzador unificado (SLURM) para ambos modelos
│
├── VideoMAEv2_Proyecto_Paper/     # Análisis de VideoMAEv2 (Luis)
│   ├── eval_ssv2.sh               # Evaluación en Something-Something V2
│   ├── eval_k400.sh               # Evaluación en Kinetics-400
│   ├── eval_checkpoint.sh         # Evaluación de checkpoints personalizados
│   ├── eval_extracted_features.sh # Evaluación con features pre-extraídos
│   ├── extract_tad_feature.py     # Extracción de features para TAD (THUMOS14, FineAction)
│   ├── ver_ultimos_features.py    # Inspección de features internos (capas profundas)
│   ├── visualizar_atencion.py     # Visualización de mapas de atención del Transformer
│   ├── ver_prediccion.py          # Interpretación de predicciones del modelo
│   ├── leer_npy.py                # Utilidad para inspeccionar archivos .npy de features
│   ├── run_class_finetuning.py    # Script de fine-tuning de clasificación
│   ├── run_mae_pretraining.py     # Script de pre-entrenamiento MAE
│   ├── models/                    # Arquitecturas ViT y VideoMAE
│   ├── checkpoints/               # Pesos del modelo descargados (ViT-G, K710, SSV2, etc.)
│   ├── dataset/                   # Datos locales (SSV2, K400) y etiquetas CSV
│   └── logs/                      # Salidas de jobs SLURM
│
└── vjepa2/                        # Análisis de V-JEPA 2 (Renzo)
    ├── src/                       # Código fuente del modelo V-JEPA 2
    ├── evals/                     # Scripts de evaluación
    ├── configs/                   # Configuraciones YAML
    └── notebooks/                 # Notebooks de análisis
```

---

## 🧠 Modelos Analizados

### VideoMAEv2 (Luis Izaguirre)

[VideoMAEv2](https://arxiv.org/abs/2303.16727) (CVPR 2023) es un escalado del paradigma de Video Masked Autoencoders usando **Dual Masking** (máscara de tubo + máscara decodificador). El modelo estudiado es el **ViT-Giant** pre-entrenado en Kinetics-710 y fine-tuneado tanto en **SSV2** como en **K400**.

Se han realizado las siguientes actividades sobre este modelo:
- ✅ Evaluación de rendimiento en SSV2 y Kinetics-400 en Khipu.
- ✅ Extracción de features internos (CLS token y tokens de parches espacio-temporales).
- ✅ Visualización de mapas de atención por capa del Transformer mediante *forward hooks*.
- ✅ Análisis de la distribución de representaciones internas (`features_extraidos.npy`).

### V-JEPA 2 (Renzo Felix)

[V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2/) es el modelo de predicción de video de Meta AI basado en la arquitectura Joint Embedding Predictive. El análisis de Renzo sobre este modelo sigue la misma metodología de extracción de features e interpretabilidad.

---

## 🚀 Uso

### Requisitos

- Python ≥ 3.9
- PyTorch ≥ 2.0 con soporte CUDA
- `decord`, `torchvision`, `timm`, `numpy`, `opencv-python`, `matplotlib`
- Acceso al clúster Khipu con las particiones `gpu` y `data-science`

Instalar dependencias (entorno `videomae_luis_izaguirre`):

```bash
pip install -r VideoMAEv2_Proyecto_Paper/requirements.txt
pip install decord
```

### Lanzador Unificado

El `main.py` en la raíz del proyecto sirve como punto de entrada único para enviar evaluaciones a SLURM:

```bash
# Evaluar VideoMAEv2 en SSV2
python main.py --model videomae --dataset ssv2

# Evaluar VideoMAEv2 en Kinetics-400
python main.py --model videomae --dataset k400

# Evaluar V-JEPA 2 (variante ViT-H) en SSV2
python main.py --model vjepa --vjepa_variant ViT-H --dataset ssv2
```

### Evaluación directa con SLURM

```bash
# Desde VideoMAEv2_Proyecto_Paper/
sbatch eval_ssv2.sh    # Evaluación en Something-Something V2
sbatch eval_k400.sh    # Evaluación en Kinetics-400
```

### Extracción de Features Internos

```bash
cd VideoMAEv2_Proyecto_Paper/
python ver_ultimos_features.py
# Guarda las activaciones en features_extraidos.npy
```

### Inspección de Features

```bash
python leer_npy.py
# Muestra dimensiones, estadísticas y tipo de representación (CLS vs. patch tokens)
```

### Visualización de Atención

```bash
python visualizar_atencion.py
# Genera mapa_atencion_<accion>.jpg con el heatmap superpuesto al frame
```

### Inspección de Predicciones

```bash
python ver_prediccion.py
# Lee los logits del archivo de resultados y traduce el índice al nombre de acción
```

---

## 📊 Datasets

| Dataset | # Clases | Split evaluado | Ruta en Khipu |
|---|---|---|---|
| Something-Something V2 (SSV2) | 174 | Validación | `dataset/ssv2_luis/` |
| Kinetics-400 (K400) | 400 | Validación | `/home/datasets/k400/val` |
| Kinetics-710 (K710) | 710 | Pre-entrenamiento | (checkpoint externo) |

---

## 🏗️ Infraestructura

Los experimentos se ejecutan en el clúster **Khipu** de UTEC utilizando el sistema de gestión de trabajos **SLURM**:

| Parámetro | SSV2 | K400 |
|---|---|---|
| Partición | `gpu` | `data-science` |
| GPU | NVIDIA (shard) | NVIDIA RTX A6000 (48 GB) |
| CPUs | 8 | 16 |
| Memoria | 32 GB | 64 GB |
| Tiempo máx. | 4 días | 6 días |
| QoS | `a-investigacion1` | `a-investigacion1` |

---

## 📚 Referencias

```bibtex
@InProceedings{wang2023videomaev2,
    author    = {Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
    title     = {VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14549-14560}
}
```

- [VideoMAEv2 – Paper (arXiv)](https://arxiv.org/abs/2303.16727)
- [VideoMAEv2 – Repositorio oficial](https://github.com/OpenGVLab/VideoMAEv2)
- [V-JEPA 2 – Meta AI Research](https://ai.meta.com/research/vjepa/)
- [VideoMAEv2 – Pesos en Hugging Face](https://huggingface.co/OpenGVLab/VideoMAE2)

---

## 📝 Notas de Desarrollo

- Los jobs SLURM generan logs en `VideoMAEv2_Proyecto_Paper/logs/` con el formato `eval_<dataset>_<job_id>.{out,err}`.
- El entorno Conda del proyecto es `videomae_luis_izaguirre` (disponible en Khipu).
- El checkpoint principal de VideoMAEv2 usado es `vit_g_hybrid_pt_1200e_k710_ft.pth` (ViT-Giant pre-entrenado en K710).
- Se utiliza la variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para evitar fragmentación de memoria en CUDA.
