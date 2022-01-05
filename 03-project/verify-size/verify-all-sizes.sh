#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    echo "Verifies all files in the given directory have the same width and height."
    exit 1
fi

DIR="$1"

for d in "$DIR"/*; do
    echo "$d"
    ./mds-repo/noiseprint/verify-size.sh "$d"
done