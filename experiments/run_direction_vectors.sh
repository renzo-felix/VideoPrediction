#!/bin/bash
#SBATCH --job-name=direction_vectors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=ds001
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/direction_vectors_%j.log
#SBATCH --error=experiments/logs/direction_vectors_%j.err

# ============================================================
# Paso A: Calcular direction vectors en TODAS las capas (0-23)
# Modelo: V-JEPA ViT-L (vitl.pt) — 24 capas, hidden_dim=1024
# ============================================================

set -e
cd /home/renzo.felix/VideoPrediction

conda activate video

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
