#!/bin/bash
#SBATCH --job-name=vjepa2_speed_k400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ag001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=eval_speed_k400_%j.out
#SBATCH --error=eval_speed_k400_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=renzo.felix@utec.edu.pe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd /home/renzo.felix/VideoPrediction/vjepa2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paso 1: generar CSV si no existe
CSV=speed_probe/physical_diagnostics_k400.csv
if [ ! -f "$CSV" ]; then
    echo "Generando CSV de K400..."
    python speed_probe/make_k400_csv.py \
        --k400_csv data/K400/train.csv \
        --output "$CSV"
fi

# Paso 2: correr el probe
# data_root se ignora para K400 (paths absolutos en el CSV)
python -u speed_probe/run_speed_probe.py \
    --checkpoint checkpoints/vitl.pt \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --csv "$CSV" \
    --data_root /home/datasets/k400/train \
    --video_format mp4 \
    --output_dir speed_probe/output/vitl_k400 \
    --batch_size 8 \
    --num_workers 8 \
    --no_wandb
