# Authors #
# Matteo Franzil, Claudio Facchinetti, Paolo Chistè
import json
from argparse import ArgumentParser

import numpy as np
from camerafp.functions import crosscorr
from camerafp.maindir import PCE

from .logger import *
from .noisewrap import get_noiseprint_from_path


# This file was taken from the CameraIDcode repository


def crop_center(array: np.ndarray, d_width: int, d_height: int) -> np.ndarray:
    """
    Crop the array, returning the values at
    the "centre" of the array
    given by width x height.
    """
    s_height = array.shape[0]
    s_width = array.shape[1]
    if s_height < d_height or s_width < d_width:
        raise ValueError("Image is smaller than the required crop area. WTF mate.")
    else:
        # Calculate the start and end points
        start_height = (s_height - d_height) // 2
        start_width = (s_width - d_width) // 2
        end_height = start_height + d_height
        end_width = start_width + d_width
        log.debug(f"[{start_height}, {start_width}] to [{end_height}, {end_width}]")
        return array[start_height:end_height, start_width:end_width]


def normalized_cross_corr(template: np.ndarray, image: np.ndarray) -> float:
    res1_arr = np.reshape(template, [template.shape[0] * template.shape[1]])
    a: np.array = (res1_arr - np.mean(res1_arr)) / (np.std(res1_arr))
    res2_arr = np.reshape(image, [image.shape[0] * image.shape[1]])
    b: np.array = (res2_arr - np.mean(res2_arr)) / (np.std(res2_arr))
    return (1 / (image.shape[0] * image.shape[1])) * abs(np.inner(a, b))


def matrix_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Calculates the distance between two matrices
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices are not the same size. WTF mate.")
    else:
        # sum = 0
        # for i in range(matrix1.shape[0]):
        #     for j in range(matrix1.shape[1]):
        #         sum += (matrix1[i, j] - matrix2[i, j]) ** 2
        return np.linalg.norm(matrix1 - matrix2)


def validate(image: np.ndarray, template: np.ndarray, image_name: str, template_name: str):
    # Width and height are flipped
    t_height, t_width = template.shape
    i_height, i_width = image.shape

    # Obtain central crop of both image and template
    CROP_SIZE = (1024, 1024)
    if t_height < CROP_SIZE[0] or t_width < CROP_SIZE[1] or i_height < CROP_SIZE[0] or i_width < CROP_SIZE[1]:
        log.debug("Template is smaller than the required crop area. Reducing crop")
        CROP_SIZE = (512, 512)

    template = crop_center(template, CROP_SIZE[1], CROP_SIZE[0])
    image = crop_center(image, CROP_SIZE[1], CROP_SIZE[0])

    # if i_width > t_width or i_height > t_height:
    #     log.debug("Image is larger than template. Resizing image...")
    #     image = crop_array(image, t_width, t_height)
    # elif i_width < t_width or i_height < t_height:
    #     log.debug("Image is smaller than template. Resizing template...")
    #     template = crop_array(template, i_width, i_height)

    # Calculate the correlation
    log.info("Calculating correlation...")
    cross_corr = crosscorr(template, image)
    ncc = normalized_cross_corr(template, image)
    pce = PCE(cross_corr)
    dist = matrix_distance(template, image)

    return {
        "template": {
            "name": str(template_name),
            "width": int(t_width),
            "height": int(t_height)
        },
        "image": {
            "name": str(image_name),
            "width": int(i_width),
            "height": int(i_height)
        },
        "pce": pce[0]['PCE'],
        "ncc": ncc,
        "distance": float(dist)
    }


def get_data(image_path: str, template_path: str) -> {}:
    image, template = get_noiseprint_from_path(image_path, template_path)

    result = validate(
        image=image,
        template=template,
        image_name=image_path,
        template_name=template_path
    )

    return result


# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description='Noise correlation calculator')
    parser.add_argument('-t', '--template', dest='template', required=True,
                        help='Template (fingerprint) file')
    parser.add_argument('-i', '--image', dest='image', required=True,
                        help='File to be checked against')

    args = parser.parse_args()

    output = get_data(args.image, args.template)

    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()
