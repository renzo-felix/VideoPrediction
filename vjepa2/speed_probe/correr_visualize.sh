#!/bin/bash
#SBATCH --job-name=vjepa2_vis_layers
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                  # 200MB por video × 2 + PCA + matplotlib
#SBATCH --time=0-00:30:00          # Solo 1-2 videos → max 30 min
#SBATCH --output=vis_layers_%j.out
#SBATCH --error=vis_layers_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd /home/renzo.felix/VideoPrediction/vjepa2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Elige 2 videos: uno rápido y uno lento ──────────────────────────────────
# Busca un ejemplo en el CSV (ajusta los IDs manualmente si quieres otros)
VIDEO_FAST="data/ssv2/videos/20bn-something-something-v2/34899.webm"
VIDEO_SLOW="data/ssv2/videos/20bn-something-something-v2/220847.webm"

python -u speed_probe/visualize_layers.py \
    --checkpoint checkpoints/vitl.pt \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --videos "$VIDEO_FAST" "$VIDEO_SLOW" \
    --labels "Rápido" "Lento" \
    --component residual \
    --all_layers \
    --fps 3 \
    --output_dir speed_probe/output/vis
