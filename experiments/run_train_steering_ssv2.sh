#!/bin/bash
#SBATCH --job-name=train_steer_ssv2
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --output=experiments/logs/train_steer_ssv2_%j.log
#SBATCH --error=experiments/logs/train_steer_ssv2_%j.err

module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

mkdir -p experiments/results
mkdir -p experiments/logs

echo "=== TRAINING STEERING VECTOR SSV2 ==="
python experiments/src/train_steering.py \
    --model videomae \
    --concept actual_speed \
    --method percentile \
    --activations-dir experiments/activations/counterfactuals_ssv2_videomae \
    --metadata-path experiments/data/counterfactual_metadata_ssv2.csv \
    --output experiments/results/steering_vector_ssv2_videomae.pkl
echo "=== DONE ==="
