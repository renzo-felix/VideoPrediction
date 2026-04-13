#!/bin/bash
#SBATCH --job-name=probing_40layers
#SBATCH --output=logs/probing_40layers_%j.out
#SBATCH --error=logs/probing_40layers_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=2-00:00:00

# =============================================================================
# PASO 1: Extracción Exhaustiva de 40 Capas + Linear Probing + Clustering Video
# =============================================================================
# Escala la extracción de activaciones de 11 capas discretas a las 40 capas
# completas del ViT-Giant (VideoMAEv2) sobre SSv2.
# Tiempo estimado: ~16h (basado en 4h21min con 11 capas, Job 29344).
# Partición: gpu (nodos g001/g002 idle) en vez de data-science (ocupada).
# =============================================================================

# 1. CARGA DE MÓDULOS BASE
module load cuda/11.8
module load miniconda/3.0

# 2. ACTIVACIÓN DEL ENTORNO CONDA
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# 3. CONFIGURACIÓN DE RUTAS PARA TRITON (SOLUCIÓN stdlib.h)
# Estas variables apuntan a las librerías internas del entorno Conda para
# que compiladores como Triton encuentren headers y libs del sistema.
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

# 4. PREVENCIÓN DE FRAGMENTACIÓN DE VRAM (CRÍTICO PARA 40 CAPAS)
# expandable_segments permite que CUDA reutilice segmentos de memoria
# fragmentados en vez de fallar con OOM. Esencial cuando se generan
# 120 tensores de [1, 2048, 1408] por video durante 21,202 videos.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. AUTENTICACIÓN WANDB
# Extraemos la API key del archivo ~/.netrc para evitar el error de
# autenticación que ocurrió en el Job 29344 al intentar wandb.init().
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')

# 6. EJECUCIÓN DEL PIPELINE (EXTRACCIÓN + LINEAR PROBING)
# --layers: se pasan las 40 capas (0 a 39) explícitamente.
# Los archivos de salida serán:
#   - output_dir/activations_40layers.npz (~13 GB estimado)
#   - output_dir/probing_results_40layers.json
# Los archivos existentes (activations.npz, probing_results.json) NO se sobrescriben.
python -u run_layer_probing.py \
    --layers $(seq 0 39 | tr '\n' ' ') \
    --wandb_project videomae_probing \
    --output_dir output_dir

# 7. GENERACIÓN DEL VIDEO DE CLUSTERING (PCA+UMAP)
# Usa el .npz recién generado para crear una animación de clustering que
# muestra la evolución semántica capa a capa con las 40 capas completas.
# El video existente (clustering_evolution.mp4, 11 capas) NO se sobrescribe.
python -u visualize_clustering_evolution.py \
    --npz output_dir/activations_40layers.npz \
    --output_dir videos_simulation_clustering \
    --output_name clustering_evolution_ssv2_40layers.mp4

echo "✅ Pipeline completo: probing 40 capas + video de clustering SSv2"
