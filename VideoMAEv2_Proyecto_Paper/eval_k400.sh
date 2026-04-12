#!/bin/bash
#SBATCH --job-name=eval_k400
#SBATCH --output=logs/eval_k400_%j.out
#SBATCH --error=logs/eval_k400_%j.err
#SBATCH --partition=data-science
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --time=6-00:00:00               
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# 1. CARGA DE MÓDULOS BASE
# Eliminamos "module load gcc" para usar el de Conda
module load cuda/11.8
module load miniconda/3.0

# 2. ACTIVACIÓN DEL ENTORNO
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# 3. CONFIGURACIÓN DE RUTAS PARA TRITON (SOLUCIÓN stdlib.h) 
# Estas variables apuntan ahora a las librerías internas de tu entorno Conda
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export SLURM_NTASKS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# El enlace simbólico para K400 (une el prefijo del CSV con los videos reales)
ln -sfn /home/datasets/k400/val k400

# 4. EJECUCIÓN DEL BENCHMARK [cite: 132, 134, 136]
# Se mantiene el batch_size en 4 gracias a los 48GB de la RTX A6000 en ds001
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) \
    run_class_finetuning.py \
    --model vit_giant_patch14_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path dataset/k400_luis/labels \
    --data_root . \
    --finetune /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth \
    --log_dir output_dir/k400_eval \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --num_workers 2 \
    --eval \
    --dist_eval