import os
from typing import Tuple

import numpy as np
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f, jpeg_qtableinv
from scipy.io import loadmat

from .logger import *


def generate_noiseprint(image_path: str) -> np.ndarray:
    img1, mode1 = imread2f(image_path, channel=1)
    try:
        QF = jpeg_qtableinv(str(image_path))
    except:
        QF = 200
    log.debug(f"Generating noiseprint for {image_path}...")
    return genNoiseprint(img1, QF)


# def load_single_noiseprint(template_path: str) -> np.ndarray:
#     if not template_path.endswith('.mat'):
#         raise ValueError("Template must be a .mat file")
#     return loadmat(template_path)['noiseprint']


def get_noiseprint_from_path(image_path: str, template_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Verify sanity of args
    if image_path is None or template_path is None:
        raise AttributeError("Please specify a template and an image file")

    # Open the files
    try:
        if template_path.endswith('.mat'):
            template: np.ndarray = loadmat(template_path)['noiseprint']
        else:
            raise Exception("Template file must be a .mat file")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {template_path}", e)
    except ValueError as e:
        raise ValueError("Mat file does not have a noiseprint variable", e)
    except Exception as e:
        raise RuntimeError(f"Unknown error! Check exception", e)

    # First check if mat file has been already
    # generated and do it on the fly just in case
    # image_mat = f"{image.replace('.JPG', '.mat').replace('.jpg', '.mat')}"
    # if not os.path.exists(f"{image_mat}"):
    #     log.debug(f"{image_mat} does not exist, generating it IN THE SAME FOLDER (remove it later)")
    #     os.system(f'python3 noiseprint/main_extraction.py {image} {image_mat}')

    try:
        if image_path.endswith('.mat'):
            image: np.ndarray = loadmat(image_path)['noiseprint']
        else:
            # First check if a mat file was already created
            if os.path.isfile(image_path.replace('.JPG', '.mat').replace('.jpg', '.mat')):
                image: np.ndarray = loadmat(image_path.replace('.JPG', '.mat').replace('.jpg', '.mat'))['noiseprint']
            else:
                # Convert the image on the fly using noiseprint
                log.debug("Image file must be a .mat file: converting...")
                os.system(f'python3 mds/noiseprint/main_extraction.py'
                          f' {image_path} {image_path.replace(".JPG", ".mat").replace(".jpg", ".mat")}')
                image: np.ndarray = loadmat(image_path.replace(".JPG", ".mat").replace(".jpg", ".mat"))['noiseprint']
    except FileNotFoundError as e:
        possible_picture = image_path.replace(".mat", ".jpg")

        message = f"File not found: {image_path}"
        if os.path.isfile(possible_picture):
            message += "(did you mean to use the jpg file?)"

        raise FileNotFoundError(message, e)
    except ValueError as e:
        raise ValueError("Mat file does not have a noiseprint variable", e)
    except Exception as e:
        raise RuntimeError(f"Unknown error! Check exception: {e}")

    # We're ready
    log.info(f"Image: {image_path} @ {image.shape}")
    log.info(f"Template: {template_path} @ {template.shape}")

    return image, template
