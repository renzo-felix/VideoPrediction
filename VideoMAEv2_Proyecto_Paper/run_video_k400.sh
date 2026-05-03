#!/bin/bash
#SBATCH --job-name=video_k400
#SBATCH --output=logs/video_k400_%j.out
#SBATCH --error=logs/video_k400_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=g002
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=0-01:00:00

# 1. CARGA DE MÓDULOS BASE
module load cuda/11.8
module load miniconda/3.0
# ffmpeg es necesario para generar el mp4
module load ffmpeg 2>/dev/null || true

# 2. ACTIVACIÓN DEL ENTORNO CONDA
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre
conda install -y -c conda-forge ffmpeg

# 3. CONFIGURACIÓN
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

echo "[INFO] Iniciando Generación de Video UMAP para K400..."
python -u visualize_clustering_evolution.py \
    --npz output_dir/activations_k400_40layers.npz \
    --output_dir videos_simulation_clustering \
    --output_name clustering_evolution_k400.mp4

echo "[INFO] Video Generado."
