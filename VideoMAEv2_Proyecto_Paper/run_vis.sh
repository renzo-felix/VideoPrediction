#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --gres=gpu:1
#SBATCH --job-name=videomae_vis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=4-00:00:00

# ============================================================================
# run_vis.sh
# Renderiza el clustering evolutivo usando UMAP + PCA sobre activaciones previas.
# Requiere que run_probing.sh haya generado "output_dir/activations.npz".
# ============================================================================

module load cuda/11.8
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

mkdir -p logs

echo "=========================================="
echo "Clustering Animado sobre VideoMAEv2 ViT-Giant"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================="

python -u visualize_clustering_evolution.py

echo ""
echo "Script finalizado: $(date)"