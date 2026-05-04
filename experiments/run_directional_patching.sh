#!/bin/bash
#SBATCH --job-name=directional_patching
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6-00:00:00
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/directional_patching_%j.log
#SBATCH --error=experiments/logs/directional_patching_%j.err

# ============================================================
# Paso C: Directional Patching — prueba de causalidad
# Modelo: V-JEPA ViT-L (vitl.pt) — 24 capas, final-layer=23
# PRERREQUISITO: correr run_direction_vectors.sh primero.
# ============================================================

set -e
cd /home/renzo.felix/VideoPrediction

conda activate video

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p experiments/logs
mkdir -p experiments/results
mkdir -p experiments/figures

echo "=== DIRECTIONAL PATCHING — V-JEPA ViT-L ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Capas equiespaciadas dentro de las 24 capas de ViT-L (0-23)
LAYERS="0 3 6 9 12 15 18 21 23"

echo ""
echo "--- V-JEPA ViT-L: patching SLOW (videos lentos, α positivo → rápido) ---"
python experiments/src/directional_patching.py \
    --model vjepa \
    --checkpoint /home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt \
    --direction-vectors experiments/results/direction_vectors_layerwise_vjepa.pkl \
    --layers $LAYERS \
    --final-layer 23 \
    --alphas -3 -2 -1 0 1 2 3 \
    --n-videos 50 \
    --speed-group slow \
    --device cuda \
    --null-experiments \
    --output experiments/results/directional_patching_vjepa_slow.pkl

echo ""
echo "--- V-JEPA ViT-L: patching FAST (videos rápidos, α negativo → lento) ---"
python experiments/src/directional_patching.py \
    --model vjepa \
    --checkpoint /home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt \
    --direction-vectors experiments/results/direction_vectors_layerwise_vjepa.pkl \
    --layers $LAYERS \
    --final-layer 23 \
    --alphas -3 -2 -1 0 1 2 3 \
    --n-videos 50 \
    --speed-group fast \
    --device cuda \
    --null-experiments \
    --output experiments/results/directional_patching_vjepa_fast.pkl

echo ""
echo "=== DONE: directional_patching_vjepa_slow.pkl y _fast.pkl guardados ==="
echo ""
echo "Resultados clave a revisar en los logs:"
echo "  - Monotonicidad perfecta (1.0) → causalidad FUERTE"
echo "  - Monotonicidad >= 0.7        → causalidad PARCIAL"
echo "  - Monotonicidad < 0.7         → no hay evidencia causal"
echo ""
echo "Null experiments guardados en figures/null_comparison_vjepa.png"
echo "Veredicto CAUSAL = real supera random+PCA por margen > 0.2 y mono >= 0.7"
