#!/bin/bash
#SBATCH --job-name=direction_vectors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=g002
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/direction_vectors_%j.log
#SBATCH --error=experiments/logs/direction_vectors_%j.err

# ============================================================
# Paso A: Calcular direction vectors en TODAS las capas (0-39)
# ============================================================
# Prerequisito: haber generado videos y extraído activaciones en
# al menos una capa (para tener training metadata).
# Este script extrae activaciones de TODAS las capas en un solo
# forward pass por video, lo que es mucho más eficiente.

set -e
cd /home/projects/video-prediction

conda activate video

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p experiments/logs
mkdir -p experiments/results
mkdir -p experiments/figures

echo "=== DIRECTION VECTORS LAYERWISE ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# V-JEPA
echo ""
echo "--- V-JEPA ---"
python experiments/src/direction_vectors_layerwise.py \
    --model vjepa \
    --checkpoint experiments/models/vitg16.pth \
    --device cuda \
    --low-percentile 33 \
    --high-percentile 67

# VideoMAE
echo ""
echo "--- VideoMAE ---"
python experiments/src/direction_vectors_layerwise.py \
    --model videomae \
    --checkpoint experiments/models/videomae_vitg.pth \
    --device cuda \
    --low-percentile 33 \
    --high-percentile 67

echo ""
echo "=== DONE: direction_vectors_layerwise_vjepa.pkl y _videomae.pkl guardados ==="
