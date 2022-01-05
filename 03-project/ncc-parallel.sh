#!/bin/zsh

for f in "/Users/matte/Downloads/mds/fingerprints"/*; do
    if [[ $f =~ .*Canon-EOS-1200D\.mat ]]; then # Outcamera
        python3 ./mds-repo/project/ncc-all.py -t "$f" -i "/Volumes/Extreme SSD/_move/dataset-outcamera-validation-noise/" -o -H "/Volumes/Extreme SSD/_move/dataset-dresden-validation-noise" >> "$f.log" 2>&1 &
    elif [[ $f =~ .*\.mat ]]; then
        python3 ./mds-repo/project/ncc-all.py -t "$f" -i "/Volumes/Extreme SSD/_move/dataset-dresden-validation-noise" >> "$f.log" 2>&1 &
    fi
done
