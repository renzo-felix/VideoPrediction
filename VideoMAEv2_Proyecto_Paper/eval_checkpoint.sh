#!/bin/bash
#SBATCH --job-name=eval_videomae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# --- 1. LIMPIEZA Y CARGA DE MÓDULOS ---
source ~/miniconda3/etc/profile.d/conda.sh

# A) Descargar cualquier módulo incompatible
module purge

# B) Cargar el compilador compatible (GCC 9.4.0)
module load gnu9/9.4.0

# C) Cargar CUDA 11.8
module load cuda/11.8

# D) Forzar variables de entorno
export CUDA_HOME="/opt/ohpc/pub/compiler/cuda/11.8.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# --- 2. CONFIGURACIÓN DE PYTHON ---
# Usamos el entorno 'lab1' directo
PYTHON_EXEC="/home/renzo.felix/miniconda3/envs/lab1/bin/python"

# Fix para Protobuf
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Diagnóstico en el log
echo "=== CONFIGURACIÓN DE COMPILADOR ==="
echo "Compilador C++ activo:"
which g++
g++ --version | head -n 1
echo "CUDA Compiler:"
nvcc --version | head -n 4
echo "==================================="

# Arreglo para TRITON
export TRITON_CACHE_DIR="/tmp/triton_cache_$USER"
mkdir -p $TRITON_CACHE_DIR

# --- 3. RUTAS DEL PROYECTO ---
BASE_DIR="/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper"
MODEL_PATH="$BASE_DIR/checkpoints/checkpoints/vit_b_k710_dl_from_giant.pth" 
DATA_PATH="$BASE_DIR/data/k710"
OUTPUT_DIR="$BASE_DIR/work_dir/eval_test"

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# --- 4. EJECUCIÓN ---
echo "Iniciando script de evaluación..."

$PYTHON_EXEC -u run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-710 \
    --nb_classes 710 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --eval \
    --dist_eval