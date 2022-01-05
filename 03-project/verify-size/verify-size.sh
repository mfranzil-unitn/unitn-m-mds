#!/bin/bash

DIR=$1

cd "$DIR" || exit

ctr=0
width=0
height=0

for image in *; do
    exif=$(exiftool $image)
    if [[ ctr -eq 0 ]]; then
        width=$(echo "$exif" | grep -E 'Width' | grep 'Exif' | cut -d ':' -f 2 | cut -d ' ' -f 2)
        height=$(echo "$exif" | grep -E 'Height' | grep 'Exif' | cut -d ':' -f 2 | cut -d ' ' -f 2)
        echo "Base: $width x $height"
    else
        __width=$(echo "$exif" | grep -E 'Width' | grep 'Exif' | cut -d ':' -f 2 | cut -d ' ' -f 2)
        __height=$(echo "$exif" | grep -E 'Height' | grep 'Exif' | cut -d ':' -f 2 | cut -d ' ' -f 2)
        if [[ $width -ne $__width ]] || [[ $height -ne $__height ]]; then
            echo "Image $image has different dimensions than the first image"
            echo "First image: $width x $height"
            echo "Current image: $__width x $__height"
            echo "Fix this..."
            break
        fi
    fi
    ctr=$((ctr+1))
done