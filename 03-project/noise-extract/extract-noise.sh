#!/bin/bash

DIR=$1
OUTDIR=$2

cd "$DIR" || exit

for image in *; do
    echo conda run -n mds4 python3 ../../noiseprint/main_extraction.py "$image" "../../$OUTDIR/${image//\.JPG/}.mat" | bash
done