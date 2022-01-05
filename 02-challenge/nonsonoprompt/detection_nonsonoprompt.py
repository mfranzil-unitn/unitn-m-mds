import math

import cv2
import numpy as np
import pywt
from numpy import inf
from scipy import spatial
from scipy.fft import dct
from skimage.util import view_as_blocks

from .wpsnr import wpsnr

MARK_SIZE: int = 1024
ALPHA: float = 0.3
THRESHOLD: float = 62.779416765498354  # calculated with threshold_nonsonoprompt.py
BLOCK_SIZE = (32, 32)


def extract_from_subband(original_subband, watermark_subband, alpha, start, stop):
    result_size = stop - start
    result = []  # for ease of coding, the mark is stored in a plain python array. is converted
    # to a narray outside of this function
    mark_index = 0
    for r in range(0, np.shape(original_subband)[0]):
        if mark_index >= result_size:
            break
        for c in range(0, np.shape(original_subband)[1]):
            if mark_index >= result_size:
                break
            ex_mark_value = (watermark_subband[r][c] - original_subband[r][c]) / alpha
            result.append(abs(ex_mark_value))
            mark_index = mark_index + 1
    return result


def extract_wm_wavelets(image, watermarked, alpha=ALPHA):
    image_wavelet = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = image_wavelet

    watermarked_wavelet = pywt.dwt2(watermarked, 'bior1.3')
    w_LL, (w_LH, w_HL, w_HH) = watermarked_wavelet
    w_ex = []

    w_ex.extend(extract_from_subband(HH, w_HH, alpha, 0, 768))
    w_ex.extend(extract_from_subband(HL, w_HL, alpha, 768, 1024))

    return np.array(w_ex)


def texture_areas(image, block_size=BLOCK_SIZE, coeff=1.0, result=None, it=0):
    # seed: int = int.from_bytes(hashlib.sha512(str(image).encode('utf-8')).digest(), byteorder='little')
    # random.seed(seed)
    if it == 50:
        return result
    if result is None:
        result = []

    image_variance = np.var(image, ddof=1)
    # print(f"var:{image_variance * coeff}")
    (block_size_row, block_size_col) = block_size
    i = j = 0
    blocks = view_as_blocks(image, (block_size_row, block_size_col))
    blocks = [blocks[r][c] for r in range(np.shape(blocks)[0]) for c in range(np.shape(blocks)[1])]
    for block in blocks:
        var = np.var(block, ddof=1)
        if var > image_variance * coeff and (i, j, var) not in result:
            result.append((i, j, var))
        j += block_size_col
        if j == np.shape(image)[1]:
            j = 0
            i += block_size_row

    if len(result) < 6 * math.sqrt(image_variance):
        texture_areas(image, block_size, coeff / 1.15, result, it=it + 1)
        # seed: int = int.from_bytes(hashlib.sha512(str(image).encode('utf-8')).digest(), byteorder='little')
        # random.seed(seed)
    else:
        print(it)
    result.sort(key=lambda x: -x[2])
    result = [(i, j) for (i, j, _) in result]

    return result


def extract_wm(image, watermarked, alpha=ALPHA):
    # Get the locations of the most perceptually significant components
    res = image.copy().astype(float)

    texture_areas_indexes = texture_areas(res)
    mark_index = 0

    w_ex = np.zeros(MARK_SIZE, dtype=np.float64)

    coeff_per_block = MARK_SIZE // len(texture_areas_indexes) + 8

    for (row_index, col_index) in texture_areas_indexes:
        if mark_index >= MARK_SIZE:
            break

        block_ori = res[row_index:row_index + BLOCK_SIZE[0], col_index:col_index + BLOCK_SIZE[1]]
        block_wat = watermarked[row_index:row_index + BLOCK_SIZE[0], col_index:col_index + BLOCK_SIZE[1]]

        block_ori_dct = dct(dct(block_ori, axis=0, norm='ortho'), axis=1, norm='ortho')
        block_wat_dct = dct(dct(block_wat, axis=0, norm='ortho'), axis=1, norm='ortho')

        block_ori_dct = abs(block_ori_dct)
        block_wat_dct = abs(block_wat_dct)

        locations = np.argsort(-block_ori_dct, axis=None)
        locations = [(val // BLOCK_SIZE[0], val % BLOCK_SIZE[0]) for val in locations]

        for (i, j) in locations[6:coeff_per_block]:
            if mark_index < MARK_SIZE:
                w_ex[mark_index] = (block_wat_dct[i][j] - block_ori_dct[i][j]) / (alpha * block_ori_dct[i][j])
                # print(mark_index)
                # print((block_wat_dct[i][j] - block_ori_dct[i][j]) / (alpha * block_ori_dct[i][j]))
                mark_index += 1

                if mark_index >= MARK_SIZE:
                    break

    return w_ex


# def similarity(x, x_star):
#     if np.sqrt(np.sum(np.multiply(x_star, x_star))) == 0:
#         return +inf
#
#     try:
#         s = np.sum(np.multiply(x, x_star)) / np.sqrt(np.sum(np.multiply(x_star, x_star)))
#     except FloatingPointError as ex:
#         s = -inf
#
#     return 0 if s < 0 else s

def similarity(X_star, X):
    np.seterr(all='raise')
    try:
        retval = (1 - spatial.distance.cosine(X, X_star)) * 100
        return retval if retval > 0 else 0
    except Exception as ex:
        return -inf


def detection(input1, input2, input3, alpha=ALPHA):
    """
    :param input1: the string of the name of the original image
    :param input2: the string of the name of the watermarked image
    :param input3: the string of the name of the attacked image
    :return: a tuple (output1, output2): output1 is 1 whether the mark is present, 0 otherwise; output2 is the WPSNR of said image
    """
    np.seterr(all='raise')
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    # watermark_original = extract_wm_wavelets(original, watermarked)
    # watermark_attacked = extract_wm_wavelets(original, attacked)

    if THRESHOLD == -1:
        raise Exception("Please calculate thresholds first")

    watermark_original = extract_wm(original, watermarked, alpha=alpha)
    watermark_attacked = extract_wm(original, attacked, alpha=alpha)

    sim_wt = similarity(watermark_attacked, watermark_original)

    return 1 if sim_wt > THRESHOLD else 0, wpsnr(watermarked, attacked)
