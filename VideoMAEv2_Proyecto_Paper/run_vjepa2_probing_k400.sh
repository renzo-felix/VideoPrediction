#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=vjepa2_probing_k400
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=4-00:00:00

# ============================================================================
# run_vjepa2_probing_k400.sh
# Linear Probing layer-wise de V-JEPA 2 sobre Kinetics-400 con labels de velocidad.
#
# PREREQUISITO: tener physical_diagnostics_k400.csv con speed labels para K400.
# Si aún no existe, crearlo con create_physical_subset.py adaptado para K400.
#
# K400 en Khipu ya está en /home/datasets/k400/ como symlink compartido.
# Videos en formato mp4: /home/datasets/k400/train/<video_id>.mp4
#
# Enviar: sbatch run_vjepa2_probing_k400.sh
# ============================================================================

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

mkdir -p logs output_dir/vjepa2_vitl_k400

# --- Checkpoint V-JEPA 2 ViT-Large ---
CHECKPOINT="/home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt"
# Una vez movido a carpeta compartida:
# CHECKPOINT="/home/projects/video-prediction/checkpoints/vjepa2/vitl.pt"

# K400 ya está en la carpeta compartida de datasets de Khipu
DATA_ROOT="/home/datasets/k400"

# CSV con speed labels para K400 (CREAR PRIMERO con create_physical_subset_k400.py)
CSV_PATH="/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/physical_diagnostics_k400.csv"

echo "=========================================="
echo "V-JEPA 2 Layer-wise Linear Probing - K400"
echo "Modelo: ViT-Large (24 bloques, embed_dim=1024)"
echo "Dataset: Kinetics-400 (speed labels)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: $CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "=========================================="

# Verificar que el CSV de K400 existe antes de correr
if [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] No existe $CSV_PATH"
    echo "Crear primero el CSV de speed labels para K400."
    echo "Ver: create_physical_subset.py (adaptar para K400)"
    exit 1
fi

python -u run_vjepa2_layer_probing.py \
    --checkpoint "$CHECKPOINT" \
    --model_name vit_large \
    --img_size 256 \
    --num_frames 16 \
    --csv "$CSV_PATH" \
    --data_root "$DATA_ROOT" \
    --video_format mp4 \
    --output_dir output_dir/vjepa2_vitl_k400 \
    --wandb_project vjepa2_probing \
    --wandb_run_name "vjepa2_vitl_k400_speed"

echo ""
echo "Script finalizado: $(date)"
