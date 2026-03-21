#!/bin/bash
#SBATCH --partition=data-science
#SBATCH --job-name=create_subset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# ============================================================================
# run_subset.sh
# Genera physical_diagnostics.csv a partir de las etiquetas de SSv2.
# Este script NO requiere GPU — solo parsea archivos de etiquetas.
# ============================================================================

# Verificar disponibilidad del nodo antes de lanzar:
# sinfo -n ds001 -o "%N %t"

module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

# Crear directorio de logs si no existe
mkdir -p logs

echo "=========================================="
echo "Generando subconjunto diagnóstico de velocidad"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================="

python create_physical_subset.py \
    --data_root dataset/ssv2_luis \
    --output physical_diagnostics.csv \
    --splits train val

echo ""
echo "Script finalizado: $(date)"
