#!/bin/bash
#SBATCH --job-name=extraer_ssv2
#SBATCH --output=logs/extraer_%j.out
#SBATCH --error=logs/extraer_%j.err
#SBATCH --partition=standard
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00

module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate videomae_luis_izaguirre

SRC="videos"
DST="SomethingV2"
mkdir -p $DST

echo "Iniciando extracción paralela en: $SRC"

# Buscamos los videos y los pasamos a xargs para procesar 16 a la vez (-P 16)
find "$SRC" -name "*.webm" | xargs -I {} -P 16 bash -c '
    video="{}"
    filename=$(basename -- "$video")
    dirname="${filename%.*}"
    mkdir -p "SomethingV2/$dirname"
    
    # Extraer frames de forma silenciosa
    ffmpeg -i "$video" -vf scale=-1:256 -q:v 1 "SomethingV2/$dirname/img_%05d.jpg" -nostats -loglevel error < /dev/null
'

echo "Proceso finalizado."