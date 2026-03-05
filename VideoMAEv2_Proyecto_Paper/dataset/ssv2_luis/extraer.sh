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

SRC="videos"
DST="SomethingV2"

mkdir -p $DST
echo "Iniciando búsqueda en: $SRC"

# Buscamos videos. Si no encuentra nada, el log te lo dirá.
FILES=$(find "$SRC" -name "*.webm")

if [ -z "$FILES" ]; then
    echo "ERROR: No se encontraron archivos .webm en $SRC"
    exit 1
fi

echo "$FILES" | while read video; do
    filename=$(basename -- "$video")
    dirname="${filename%.*}"
    echo "Procesando: $filename"

    mkdir -p "$DST/$dirname"
    ffmpeg -i "$video" -vf scale=-1:256 -q:v 1 "$DST/$dirname/img_%05d.jpg" -nostats -loglevel error < /dev/null
done

echo "Proceso finalizado."