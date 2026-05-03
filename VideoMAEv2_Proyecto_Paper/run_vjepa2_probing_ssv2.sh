#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=vjepa2_probing_ssv2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=4-00:00:00

# ============================================================================
# run_vjepa2_probing_ssv2.sh
# Linear Probing layer-wise de V-JEPA 2 sobre SSv2 con labels de velocidad.
#
# Usa el mismo physical_diagnostics.csv de Luis (SSv2 speed proxy).
# Modelo: V-JEPA 2 ViT-Large (más liviano, corre primero).
#
# Enviar: sbatch run_vjepa2_probing_ssv2.sh
# ============================================================================

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

mkdir -p logs output_dir/vjepa2_vitl_ssv2

# Symlink para que las rutas del CSV coincidan (igual que en run_probing.sh)
ln -sfn . dataset/ssv2_luis/SomethingV2/frames

# --- Checkpoint V-JEPA 2 ViT-Large ---
# Ruta confirmada en el repo (key="target_encoder" según config k400.yaml)
CHECKPOINT="/home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt"
# Una vez que se mueva a la carpeta compartida, cambiar a:
# CHECKPOINT="/home/projects/video-prediction/checkpoints/vjepa2/vitl.pt"

echo "=========================================="
echo "V-JEPA 2 Layer-wise Linear Probing - SSv2"
echo "Modelo: ViT-Large (24 bloques, embed_dim=1024)"
echo "Dataset: Something-Something v2 (speed labels)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# Videos SSv2 en formato .webm
# physical_diagnostics.csv tiene paths tipo "SomethingV2/frames/34899"
# → el script extrae el ID (34899) y busca DATA_ROOT/34899.webm
DATA_ROOT="/home/renzo.felix/VideoPrediction/vjepa2/data/ssv2/videos/20bn-something-something-v2"

# CSV con speed labels de Luis
CSV_PATH="/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/physical_diagnostics.csv"

python -u run_vjepa2_layer_probing.py \
    --checkpoint "$CHECKPOINT" \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --csv "$CSV_PATH" \
    --data_root "$DATA_ROOT" \
    --video_format webm \
    --output_dir output_dir/vjepa2_vitl_ssv2 \
    --wandb_project vjepa2_probing \
    --wandb_run_name "vjepa2_vitl_ssv2_speed"

echo ""
echo "Script finalizado: $(date)"
