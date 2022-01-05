import os
import sys

import cv2
import numpy as np
from cv2 import resize
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt


def awgn(img, std, seed):
    print(f"[Adding AWGN with std={std}, seed={seed}]", file=sys.stderr)
    mean = 0.0
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def blur(img, sigma):
    print(f"[Adding blur with sigma={sigma}]", file=sys.stderr)
    attacked = gaussian_filter(img, sigma)
    return attacked


def jpeg_compression(img, QF):
    print(f"[Deep frying with QF={QF}]", file=sys.stderr)

    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')

    return attacked


def median(img, kernel_size):
    print(f"[Applying median with kernel={kernel_size}]", file=sys.stderr)
    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img, scale):
    print(f"[Shrinking image with scale={scale}]", file=sys.stderr)
    x, y = img.shape
    _x = int(x * scale)
    _y = int(y * scale)

    img = img.astype('float32')
    attacked = resize(img, (_x, _y))
    attacked = resize(attacked, (x, y))

    attacked = attacked.astype(int)
    return attacked


def sharpening(img, sigma, alpha):
    print(f"[Sharpening image with sigma={sigma}, alpha={alpha}]", file=sys.stderr)
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)

    return attacked
