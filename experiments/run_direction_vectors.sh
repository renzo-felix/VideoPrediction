#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=videomae_probing
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --output=./direction_vectors_%j.log
#SBATCH --error=./direction_vectors_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe


# ====================================

# ============================================================
# Paso A: Calcular direction vectors en TODAS las capas (0-23)
# Modelo: V-JEPA ViT-L (vitl.pt) — 24 capas, hidden_dim=1024
# ============================================================

set -e
cd /home/renzo.felix/VideoPrediction

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p experiments/logs
mkdir -p experiments/results
mkdir -p experiments/figures

echo "=== DIRECTION VECTORS LAYERWISE — V-JEPA ViT-L ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "--- V-JEPA ViT-L ---"
python experiments/src/direction_vectors_layerwise.py \
    --model vjepa \
    --checkpoint /home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt \
    --device cuda \
    --low-percentile 33 \
    --high-percentile 67

echo ""
echo "=== DONE: direction_vectors_layerwise_vjepa.pkl guardado ==="
