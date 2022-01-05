import cv2
import numpy as np
import pywt
from scipy.fft import dct, idct

from .detection_nonsonoprompt import texture_areas, ALPHA, MARK_SIZE, BLOCK_SIZE


def embed_into_subband(wav_subband, mark_elements, alpha):
    mark_end = len(mark_elements)
    mark_index = 0
    modified_subband = np.zeros(np.shape(wav_subband))
    for r in range(0, np.shape(wav_subband)[0]):
        if mark_index >= mark_end:
            break
        for c in range(0, np.shape(wav_subband)[1]):
            if mark_index >= mark_end:
                break
            mark_element = mark_elements[mark_index]
            modified_subband[r][c] = wav_subband[r][c] + alpha * mark_element
            mark_index = mark_index + 1
    return modified_subband


def embedding_wavelets(input1, input2, alpha=ALPHA):
    """
    :param input1: the name of the path of the image that will be watermarked
    :param input2: the name of the path of the watermark
    :return: a tuple (mark, watermarked): the mark that has been loaded from memory and the watermarked image
    """
    image = cv2.imread(input1, 0)
    res_image = image.copy()
    mark = np.load(input2)
    image_wavelet = pywt.dwt2(image, 'bior1.3')
    # assign wavelets components into different variables, to make handling easier
    LL, (LH, HL, HH) = image_wavelet

    # we insert a large part of the mark in the HH subband, and another part
    # in the HL subband. This should make the mark more resistant.
    # code is setup so if we want, we can repeat parts of the mark in the subbands.
    # of course, the detection needs to be modified accordingly
    HH = embed_into_subband(HH, mark[0:768], alpha)
    HL = embed_into_subband(HL, mark[768:], alpha)

    image = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')
    return mark, image


def embedding(input1, input2, alpha=ALPHA, draw_mode=False):
    """
    :param input1: the name of the path of the image that will be watermarked
    :param input2: the name of the path of the watermark
    :return: a tuple (mark, watermarked): the mark that has been loaded from memory and the watermarked image
    """
    image = cv2.imread(input1, 0)
    res = image.copy().astype(float)
    mark = np.load(input2)

    if draw_mode:
        __image = image.copy().astype(float)
        __image[0:512, 0:512] = 0
    else:
        __image = None

    # Get the DCT transform of the image
    texture_areas_indexes = texture_areas(image)

    mark_index = 0

    coeff_per_block = MARK_SIZE // len(texture_areas_indexes) + 8

    for (row_index, col_index) in texture_areas_indexes:
        if mark_index < MARK_SIZE:
            block = image[row_index:row_index + BLOCK_SIZE[0], col_index:col_index + BLOCK_SIZE[1]]

            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            sign = np.sign(block_dct)
            block_dct = abs(block_dct)
            new_blk = block_dct.copy()

            locations = np.argsort(-block_dct, axis=None)
            locations = [(val // BLOCK_SIZE[0], val % BLOCK_SIZE[0]) for val in locations]

            for (i, j) in locations[6:coeff_per_block]:
                new_blk[i][j] *= 1 + (alpha * mark[mark_index])
                mark_index += 1

                if mark_index >= MARK_SIZE:
                    break

            new_blk *= sign
            new_block = idct(idct(new_blk, axis=1, norm='ortho'), axis=0, norm='ortho')
            new_block = np.rint(new_block)

            if draw_mode:
                __image[row_index:row_index + BLOCK_SIZE[0], col_index:col_index + BLOCK_SIZE[1]] = 255
            else:
                res[row_index:row_index + BLOCK_SIZE[0], col_index:col_index + BLOCK_SIZE[1]] = new_block

    if draw_mode:
        return mark, cv2.addWeighted(__image, 0.5, res, 0.5, 0.0)
    else:
        res = res.astype(int)
        return mark, res
