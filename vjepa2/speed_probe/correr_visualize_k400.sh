#!/bin/bash
#SBATCH --job-name=vjepa2_vis_k400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ag001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=vis_layers_k400_%j.out
#SBATCH --error=vis_layers_k400_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd /home/renzo.felix/VideoPrediction/vjepa2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Un video rápido y uno lento de K400 — ajusta los paths si quieres otros
VIDEO_FAST="/home/datasets/k400/train/running/$(ls /home/datasets/k400/train/running/ | head -1)"
VIDEO_SLOW="/home/datasets/k400/train/reading_book/$(ls /home/datasets/k400/train/reading_book/ | head -1)"

python -u speed_probe/visualize_layers.py \
    --checkpoint checkpoints/vitl.pt \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --videos "$VIDEO_FAST" "$VIDEO_SLOW" \
    --labels "Rápido (running)" "Lento (reading)" \
    --component residual \
    --all_layers \
    --fps 3 \
    --output_dir speed_probe/output/vis_k400
