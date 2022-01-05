#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 [jpg-directory]"
    echo "extract-all-noises - Extracts all noises from all jpg files in the given directory."
    echo "Files are saved in a parallel directory called [jpg-directory]-noise."
    exit 1
fi

DIR=$1

for d in "$DIR"/*; do
    echo "$d"
    outdir="${d//$DIR/$DIR-noise}"
    mkdir -p "$outdir"
    ./mds-repo/noiseprint/extract-noise.sh "$d" "$outdir"
done