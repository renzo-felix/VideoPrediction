#!/bin/bash
#SBATCH --job-name=down_k400
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # 8 cores para los 8 workers
#SBATCH --mem=8G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate videomae_luis_izaguirre

cd ~/Luis/VideoMAEv2_Proyecto_Paper/dataset/k400_luis
python descargar_k400.py
