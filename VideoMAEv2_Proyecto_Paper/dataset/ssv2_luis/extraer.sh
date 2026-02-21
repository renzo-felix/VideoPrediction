#!/bin/bash
# Buscamos los videos (tar suele crear esta subcarpeta "20bn-something-something-v2" al extraer)
SRC="video/20bn-something-something-v2"
DST="SomethingV2"

mkdir -p $DST

# Bucle para extraer frames de cada video
for video in $SRC/*.webm; do
    filename=$(basename -- "$video")
    dirname="${filename%.*}"

    # Creamos una subcarpeta para cada video dentro de "SomethingV2/"
    mkdir -p "$DST/$dirname"

    # Convertimos el video a imagenes img_00001.jpg, img_00002.jpg, etc.
    ffmpeg -i "$video" -vf scale=-1:256 -q:v 1 "$DST/$dirname/img_%05d.jpg" < /dev/null
done