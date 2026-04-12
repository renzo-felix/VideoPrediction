#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=probing_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# run_probing_test.sh
# TEST RÁPIDO: ejecuta Linear Probing con solo 100 videos.
# Para verificar que el pipeline funciona antes del run completo.
# ============================================================================

# Verificar disponibilidad del nodo antes de lanzar:
# sinfo -n ds001 -o "%N %t"

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

mkdir -p logs output_dir
ln -sfn . dataset/ssv2_luis/SomethingV2/frames

CHECKPOINT="/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth"

echo "=========================================="
echo "TEST RÁPIDO: Linear Probing (100 videos)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

python run_layer_probing.py \
    --checkpoint "$CHECKPOINT" \
    --csv physical_diagnostics.csv \
    --data_root dataset/ssv2_luis \
    --num_frames 16 \
    --max_videos 100 \
    --output_dir output_dir \
    --no_wandb \
    --layers 0 8 16 24 32 39

echo ""
echo "=========================================="
echo "TEST RÁPIDO finalizado: $(date)"
echo "=========================================="
