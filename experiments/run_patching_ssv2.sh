#!/bin/bash
#SBATCH --job-name=patch_ssv2
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/patch_ssv2_%j.log
#SBATCH --error=experiments/logs/patch_ssv2_%j.err

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== CAUSAL INTERVENTION: DIRECTIONAL PATCHING SSV2 ==="
python experiments/src/directional_patching.py \
    --model videomae \
    --checkpoint /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
    --direction-vectors experiments/results/layerwise_vector_ssv2_videomae.pkl \
    --metadata experiments/data/counterfactual_metadata_ssv2.csv \
    --layers 9 \
    --final-layer 9 \
    --alphas -10 -5 -2 0 2 5 10
echo "=== DONE ==="
