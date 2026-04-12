#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# --- 1. LIMPIEZA Y CARGA DE MÓDULOS ---
source ~/miniconda3/etc/profile.d/conda.sh
module purge
module load gnu9/9.4.0
module load cuda/11.8

# Variables de entorno para CUDA
export CUDA_HOME="/opt/ohpc/pub/compiler/cuda/11.8.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# --- 2. CONFIGURACIÓN DE PYTHON ---
PYTHON_EXEC="/home/renzo.felix/miniconda3/envs/lab1/bin/python"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== EXTRAYENDO FEATURES ==="
date

# Cache para Triton
export TRITON_CACHE_DIR="/tmp/triton_cache_$USER"
mkdir -p $TRITON_CACHE_DIR

# --- 3. CAMBIAR AL DIRECTORIO DEL PROYECTO (CRÍTICO) ---
# Sin esto, Python no encuentra 'modeling_finetune.py'
echo "Cambiando directorio a: /home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper"
cd /home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper

# --- 4. EJECUCIÓN ---
# Ejecutamos el script con el NUEVO nombre que le pusiste
echo "Ejecutando: ver_ultimos_features.py"
$PYTHON_EXEC ver_ultimos_features.py