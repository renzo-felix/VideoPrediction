#!/bin/bash
#SBATCH --job-name=vjepa2_speed_ssv2
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=eval_speed_ssv2_%j.out
#SBATCH --error=eval_speed_ssv2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd /home/renzo.felix/VideoPrediction/vjepa2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u speed_probe/run_speed_probe.py \
    --checkpoint checkpoints/vitl.pt \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --csv /home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/physical_diagnostics.csv \
    --data_root data/ssv2/videos/20bn-something-something-v2 \
    --video_format webm \
    --output_dir speed_probe/output/vitl_ssv2 \
    --wandb_project vjepa2_speed_probe \
    --wandb_run vjepa2_vitl_ssv2
