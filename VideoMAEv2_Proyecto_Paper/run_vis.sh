#!/bin/bash
#SBATCH --job-name=visualizar_attn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vis_%j.out
#SBATCH --error=logs/vis_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
module purge
module load gnu9/9.4.0
module load cuda/11.8

export CUDA_HOME="/opt/ohpc/pub/compiler/cuda/11.8.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

PYTHON_EXEC="/home/renzo.felix/miniconda3/envs/lab1/bin/python"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR="/tmp/triton_cache_$USER"
mkdir -p $TRITON_CACHE_DIR

# --- CRÍTICO: Ir a la carpeta ---
cd /home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper

echo "Iniciando visualización..."
$PYTHON_EXEC visualizar_atencion.py