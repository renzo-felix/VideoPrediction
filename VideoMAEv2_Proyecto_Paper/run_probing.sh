#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=videomae_probing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=4-00:00:00

# ============================================================================
# run_probing.sh
# Ejecuta Linear Probing sobre activaciones de VideoMAEv2 (ViT-Giant).
# Requiere GPU RTX A6000 (48GB VRAM) en el nodo data-science.
# ============================================================================

# Verificar disponibilidad del nodo antes de lanzar:
# sinfo -n ds001 -o "%N %t"

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# Solucionar fragmentación de memoria en ViT-Giant (40 bloques × 1408 dim)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Solucionar errores de compilación JIT de Triton (stdlib.h no encontrado)
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

# Crear directorio de logs y output si no existen
mkdir -p logs output_dir

# Crear symlink para que las rutas del CSV coincidan con la estructura real
ln -sfn . dataset/ssv2_luis/SomethingV2/frames

CHECKPOINT="/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth"

echo "=========================================="
echo "Linear Probing sobre VideoMAEv2 ViT-Giant"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

python -u run_layer_probing.py \
    --checkpoint "$CHECKPOINT" \
    --csv physical_diagnostics.csv \
    --data_root dataset/ssv2_luis \
    --num_frames 16 \
    --output_dir output_dir \
    --wandb_project videomae_probing \
    --layers 0 4 8 12 16 20 24 28 32 36 39

echo ""
echo "Script finalizado: $(date)"
