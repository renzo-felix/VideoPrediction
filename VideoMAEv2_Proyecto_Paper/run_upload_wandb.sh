#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=wandb_upload
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=investigacion1
#SBATCH --qos=a-investigacion1
#SBATCH --time=00:30:00

# ============================================================================
# run_upload_wandb.sh
# Ejecuta el módulo de subida a W&B mediante la partición "debug".
# Útil porque solo lee un JSON y envía una petición HTTP, toma menos de 1 minuto.
# ============================================================================

module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

mkdir -p logs

echo "=========================================="
echo "Generando sesión de W&B en Modo Offline"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================="

# 1. Habilitamos el modo Offline para evadir el bloqueo de Khipu
export WANDB_MODE=offline

# 2. Corremos el script normalmente (se guardará en /wandb local)
python -u upload_wandb.py --json_file output_dir/probing_results.json

echo ""
echo "Script finalizado: $(date)"
echo "=========================================="
echo "¡IMPORTANTE! Instrucciones de Sincronización:"
echo "Dado que corrimos esto de forma Offline, Khipu guardó tu Dashboard"
echo "en la carpeta local 'wandb/latest-run'."
echo ""
echo "Para subirlo al internet:"
echo "1. Abre tu terminal conectada al nodo maestro de Khipu."
echo "2. Asegúrate de tener activado: conda activate videomae_luis_izaguirre"
echo "3. Ejecuta el siguiente comando:"
echo "   wandb sync wandb/latest-run"
echo "=========================================="
