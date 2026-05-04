#!/bin/bash
#SBATCH --job-name=extract_ssv2
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/extract_ssv2_%j.log
#SBATCH --error=experiments/logs/extract_ssv2_%j.err

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p experiments/logs

echo "=== EXTRACTING ACTIVATIONS FOR SSV2 COUNTERFACTUALS ==="
python experiments/src/extract_activations.py \
    --mode counterfactuals \
    --dataset_name ssv2 \
    --model videomae \
    --checkpoint /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
    --device cuda
echo "=== DONE ==="
