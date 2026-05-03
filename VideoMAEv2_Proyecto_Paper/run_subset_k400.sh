#!/bin/bash
#SBATCH --job-name=probing_k400
#SBATCH --output=logs/probing_k400_%j.out
#SBATCH --error=logs/probing_k400_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodelist=g002
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=2-00:00:00

# 1. CARGA DE MÓDULOS BASE
module load cuda/11.8
module load miniconda/3.0

# 2. ACTIVACIÓN DEL ENTORNO CONDA
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# 3. CONFIGURACIÓN DE RUTAS PARA TRITON Y MEMORIA
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. AUTENTICACIÓN WANDB
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')

# 5. PASO A: Crear subset físico de K400
echo "[INFO] Iniciando Creación de Subset Físico..."
python -u create_physical_subset_k400.py \
    --csv dataset/k400_luis/labels/val.csv \
    --label_map label_map_k710.txt \
    --output physical_diagnostics_k400.csv

# 6. PASO B: Ejecutar probing en K400
echo "[INFO] Iniciando Layer Probing sobre K400..."
python -u run_layer_probing.py \
    --checkpoint /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth \
    --csv physical_diagnostics_k400.csv \
    --data_root . \
    --video_format mp4 \
    --num_classes 710 \
    --dataset_name k400 \
    --wandb_project videomae_probing_k400 \
    --output_dir output_dir

# 7. PASO C: Generar video de clustering (PCA+UMAP) para K400
echo "[INFO] Iniciando Generación de Video UMAP..."
python -u visualize_clustering_evolution.py \
    --npz output_dir/activations_k400_40layers.npz \
    --output_dir videos_simulation_clustering \
    --output_name clustering_evolution_k400.mp4

echo "[INFO] Pipeline K400 Completado Exitosamente."
