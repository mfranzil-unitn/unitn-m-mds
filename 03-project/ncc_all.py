# Authors #
# Matteo Franzil, Claudio Facchinetti, Paolo Chistè
import json
import os
import random
import sys
from argparse import ArgumentParser

from pkg import ncc
from pkg.logger import *


def do_stuff(image_folder_path, template_path, outcamera=False, outcamera_h0_image_folder_path=None):
    log.info("Executing ncc-all with the following arguments:")
    log.info(f"> image_path: {image_folder_path}")
    log.info(f"> template_path: {template_path}")

    if outcamera:
        if outcamera_h0_image_folder_path is None:
            log.error("You must specify the H0 (other cameras) folder path for outcamera validation")
            sys.exit(1)
        log.info("Running outcamera validation")
        log.info(f"> outcamera_h0_image_folder_path: {outcamera_h0_image_folder_path}")

    template: str = template_path.split('/')[-1]

    h1 = []
    h0 = []

    if outcamera:
        # Outcamera * 5
        # h1_out = match(fingerprint_1200d, immagini_fp_1200d) -> 800/5 circa
        # h0_out = match(fingerprint_1200d, rand(altre immagini non di tutti i dataset)) -> 800/5 circa
        for subfolder in os.listdir(image_folder_path):
            if not os.path.isdir(f"{image_folder_path}/{subfolder}"):
                continue

            if subfolder == '.DS_Store' or subfolder == '._.DS_Store':
                continue

            for image in os.listdir(f"{image_folder_path}/{subfolder}"):
                image_path = f"{image_folder_path}/{subfolder}/{image}"
                h1.append((image_path, image))

        for subfolder in os.listdir(f"{outcamera_h0_image_folder_path}"):
            if not os.path.isdir(f"{outcamera_h0_image_folder_path}/{subfolder}"):
                continue

            if subfolder == '.DS_Store' or subfolder == '._.DS_Store':
                continue

            for image in os.listdir(f"{outcamera_h0_image_folder_path}/{subfolder}"):
                image_path = f"{outcamera_h0_image_folder_path}/{subfolder}/{image}"
                h0.append((image_path, image))

    else:
        # for fingerprint in dresden (- Canon) * 27
        # h1 = match(fingerprint, immagini della fingerprint) -> 30
        # h0 = match(fingerprint, rand(altre immagini) -> 30
        for subfolder in os.listdir(image_folder_path):
            if not os.path.isdir(f"{image_folder_path}/{subfolder}"):
                continue

            if subfolder == '.DS_Store' or subfolder == '._.DS_Store':
                continue
            for image in os.listdir(f"{image_folder_path}/{subfolder}"):
                image_path = f"{image_folder_path}/{subfolder}/{image}"

                if subfolder == template.replace(".mat", ""):
                    h1.append((image_path, image))
                else:
                    h0.append((image_path, image))

    h0 = random.sample(h0, len(h1))

    log.info(f"Template: {template}")
    log.info(f"Images in h1: ({len(h1)})")
    for image in h1:
        log.info(f"> {image[1]}")
    log.info(f"Images in h0: ({len(h0)})")
    for image in h0:
        log.info(f"> {image[1]}")

    complete_json = {
        "h0": {},
        "h1": {},
    }

    complete_json_location = f"{template.replace('.mat', '')}.json"

    log.info("==========================================================")
    log.info("Starting NCC calculation")
    log.info(f"H1: {', '.join(item[1] for item in h1)}")
    log.info(f"H0: {', '.join(item[1] for item in h0)}")

    # H1
    log.info("(H1) ===============================================================")
    for tuple in h1:
        image_path, image = tuple

        log.info(f"Executing ncc-all on pair: {image} @ {template}")
        try:
            result = ncc.get_data(image_path, template_path)
            # Logging
            log.info(f"{image} @ {template} - {result['ncc']}")
            log.debug(f"Result: {json.dumps(result)}")
        except Exception as e:
            log.error(f"Error while processing {image} @ {template}")
            log.error(e)
            sys.exit(1)

        # Append result to json
        complete_json['h1'][image] = result
    log.info("(H1 done) ==========================================================")

    # H0
    log.info("(H0) ===============================================================")
    for tuple in h0:
        image_path, image = tuple

        log.info(f"Executing ncc-parallel on pair: {image} @ {template}")

        try:
            result = ncc.get_data(image_path, template_path)

            # Logging
            log.info(f"{image} @ {template} - {result['ncc']}")
            log.debug(f"Result: {json.dumps(result)}")
        except Exception as e:
            log.error(f"Error while processing {image} @ {template}")
            log.error(e)
            sys.exit(1)

        # Append result to json
        complete_json['h0'][image] = result
    log.info("(H0 done) ==========================================================")

    # Dump the complete JSON log
    with open(complete_json_location, 'w') as f:
        json.dump(complete_json, f, indent=4)


def entrypoint():
    parser = ArgumentParser(description='Noise correlation calculator')
    parser.add_argument('-i', '--image-folder', dest='image_folder', required=True,
                        help='Folder of subfolders containing validation images')
    parser.add_argument('-t', '--template', dest='template', required=True,
                        help='Template to be validated')
    parser.add_argument('-o', '--outcamera', dest='outcamera', action='store_true',
                        help='If set, the script will execute in outcamera mode')
    parser.add_argument('-H', '--outcamera_image_folder_path', dest='outcamera_h0_folder',
                        help='Folder of subfolders containing validation images for outcamera H0')

    args = parser.parse_args()

    if args.outcamera:
        do_stuff(
            image_folder_path=args.image_folder,
            template_path=args.template,
            outcamera=True,
            outcamera_h0_image_folder_path=args.outcamera_h0_folder
        )
    else:
        do_stuff(
            image_folder_path=args.image_folder,
            template_path=args.template
        )


if __name__ == "__main__":
    do_stuff(
        image_folder_path="/Volumes/Extreme SSD/_move/dataset-outcamera-fp-noise/",
        template_path="/Users/matte/Downloads/mds/fingerprints/Canon-EOS-1200D.mat",
            outcamera=True,
            outcamera_h0_image_folder_path="/Volumes/Extreme SSD/_move/dataset-dresden-validation-noise/"
    )
    exit(1)
    entrypoint()
