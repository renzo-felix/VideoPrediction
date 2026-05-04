#!/bin/bash
#SBATCH --job-name=directional_patching
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=g002
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/directional_patching_%j.log
#SBATCH --error=experiments/logs/directional_patching_%j.err

# ============================================================
# Paso C: Directional Patching — prueba de causalidad
# ============================================================
# PRERREQUISITO: correr run_direction_vectors.sh primero.
# Este script agrega alpha * d_L a las activaciones en capas
# seleccionadas y mide si la representación de velocidad cambia
# monotónicamente → evidencia causal.

set -e
cd /home/projects/video-prediction

conda activate video

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p experiments/logs
mkdir -p experiments/results
mkdir -p experiments/figures

echo "=== DIRECTIONAL PATCHING ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Capas a intervenir: subset equiespaciado de 8 capas representativas
LAYERS="0 5 10 15 20 25 30 35 39"

# V-JEPA
echo ""
echo "--- V-JEPA: patching en capas $LAYERS ---"
python experiments/src/directional_patching.py \
    --model vjepa \
    --checkpoint experiments/models/vitg16.pth \
    --direction-vectors experiments/results/direction_vectors_layerwise_vjepa.pkl \
    --layers $LAYERS \
    --final-layer 39 \
    --alphas -3 -2 -1 0 1 2 3 \
    --n-videos 50 \
    --speed-group slow \
    --device cuda \
    --null-experiments

# VideoMAE
echo ""
echo "--- VideoMAE: patching en capas $LAYERS ---"
python experiments/src/directional_patching.py \
    --model videomae \
    --checkpoint experiments/models/videomae_vitg.pth \
    --direction-vectors experiments/results/direction_vectors_layerwise_videomae.pkl \
    --layers $LAYERS \
    --final-layer 39 \
    --alphas -3 -2 -1 0 1 2 3 \
    --n-videos 50 \
    --speed-group slow \
    --device cuda \
    --null-experiments

echo ""
echo "=== DONE: directional_patching_vjepa.pkl y _videomae.pkl guardados ==="
echo ""
echo "Resultados clave a revisar en los logs:"
echo "  - Monotonicidad perfecta (1.0) → causalidad FUERTE"
echo "  - Monotonicidad >= 0.7        → causalidad PARCIAL"
echo "  - Monotonicidad < 0.7         → no hay evidencia causal"
echo ""
echo "Null experiments guardados en figures/null_comparison_{vjepa,videomae}.png"
echo "Veredicto CAUSAL = real supera random+PCA por margen > 0.2 y mono >= 0.7"
