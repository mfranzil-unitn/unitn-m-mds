import random

import cv2
import numpy as np
from sklearn.metrics import roc_curve

from .attacks.attacks import awgn, blur, sharpening, median, resizing, jpeg_compression
from .detection_nonsonoprompt import similarity, MARK_SIZE, extract_wm_wavelets
from .embedding_nonsonoprompt import embedding_wavelets


def random_attack(img):
    i = random.randint(1, 6)
    if i == 1:
        attacked = awgn(img, 5.0, 123)
    elif i == 2:
        attacked = blur(img, [3, 2])
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, [3, 5])
    elif i == 5:
        attacked = resizing(img, 0.5)
    elif i == 6:
        attacked = jpeg_compression(img, 75)
    else:
        raise Exception("Please provide a proper i")
    return attacked


def generate_scores_labels(impath, mark_path, scores=None, labels=None):
    image = cv2.imread(impath, 0)
    mark, watermarked = embedding_wavelets(impath, mark_path)

    if scores is None:
        scores = []

    if labels is None:
        labels = []

    sample = 0
    while sample < 999:
        if not sample % 30:
            print(f"============================\n"
                  f"Currently at sample {sample}\n"
                  f"============================")

        # fakemark is the watermark for H0
        fakemark = np.random.uniform(0.0, 1.0, MARK_SIZE)
        fakemark = np.uint8(np.rint(fakemark))
        # random attack to watermarked image
        res_att = random_attack(watermarked)
        # extract attacked watermark
        w_ex = extract_wm_wavelets(image, res_att)
        # compute similarity H1
        scores.append(similarity(mark, w_ex))
        labels.append(1)
        # compute similarity H0
        scores.append(similarity(fakemark, w_ex))
        labels.append(0)
        sample += 1

    return scores, labels


def compute_threshold(scores, labels):
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    idx_tpr = np.where((fpr - 0.1) == min(i for i in (fpr - 0.1) if i > 0))

    return tau[idx_tpr[0][0]]
