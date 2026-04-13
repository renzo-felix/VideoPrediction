#!/bin/bash
#SBATCH --job-name=Vj_k400-ViTL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ag001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=144:00:00
#SBATCH --output=eval_vitl_k400_%j.out
#SBATCH --error=eval_vitl_k400_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd /home/renzo.felix/VideoPrediction/vjepa2

python -m evals.main \
  --fname configs/eval/vitl/k400.yaml \
  --devices cuda:0
