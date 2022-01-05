import os
import random
import sys
import time

import cv2
import numpy as np

from challenge.nonsonoprompt import detection_nonsonoprompt as mod
from challenge.nonsonoprompt.attacks.attacks import blur, awgn, resizing, jpeg_compression, median, sharpening
from challenge.nonsonoprompt.detection_nonsonoprompt import THRESHOLD, extract_wm, similarity
from challenge.nonsonoprompt.embedding_nonsonoprompt import embedding
from challenge.nonsonoprompt.threshold_nonsonoprompt import generate_scores_labels, compute_threshold
from challenge.nonsonoprompt.wpsnr import wpsnr


def check_mark(x, x_star):
    x_star = np.rint(abs(x_star)).astype(int)
    res = [1 for a, b in zip(x, x_star) if a == b]
    return res


def desperation_test():
    images = [
        'buildings', 'rollercoaster', 'tree'
    ]

    mark_name = 'nonsonoprompt.npy'
    group_name = 'nonsonoprompt'
    ext = 'bmp'
    x = np.load(mark_name)

    for image_name in images:
        ori_path = '%s.%s' % (image_name, ext)
        wat_path = '%s_%s.%s' % (image_name, group_name, ext)
        ori_image = cv2.imread(ori_path, 0)
        watermarked_im = cv2.imread(wat_path, 0)

        attacked = []

        for BLR in [1, 2, 4, 8, 10, 12, 15]:  # blur
            attacked.append((blur(watermarked_im, [BLR, BLR]), 'blur', BLR))
        for STD in [2, 5, 13, 17, 23, 29]:  # awgn
            attacked.append((awgn(watermarked_im, STD, random.randint(0, 10000)), 'awgn', STD))
        for RES in [0.9, 0.7, 0.5, 0.4, 0.3, 0.2]:  # resizing
            attacked.append((resizing(watermarked_im, RES), 'res', RES))
        for QF in [90, 85, 80, 70, 50, 20, 10, 5]:  # jpeg
            attacked.append((jpeg_compression(watermarked_im, QF), 'jpeg', QF))
        for KER in [1, [1, 3], 3, [3, 5], [5, 3], 5]:  # median
            attacked.append((median(watermarked_im, KER), 'med', KER))
        for SIG in [1, 1.33, 1.5, 1.66, 2]:  # sharpening
            attacked.append((sharpening(watermarked_im, SIG, 1), 'sharp', SIG))

        for item in attacked:
            photo, attack, parameter = item
            x_star = mod.extract_wm(ori_image, photo)
            sim_wt = mod.similarity(x, x_star)
            __wp = wpsnr(ori_image, photo)
            res = check_mark(x, x_star)

            print(f"{attack};{parameter};{__wp};{sim_wt};"
                  f"{'att' if sim_wt < THRESHOLD else 'no'};{sum(res)}".replace(";", "\t\t"))


def alpha_attack_test():
    images = [
        'buildings', 'rollercoaster', 'tree'
    ]

    image_name = images[0]
    mark_path = "nonsonoprompt.npy"
    ext = 'bmp'

    ori_path = 'competition-day/%s.%s' % (image_name, ext)
    ori_image = cv2.imread(ori_path, 0)

    # 0.224, 0.241
    for attack in ['jpeg']:
        # for attack in ['awgn', 'median', 'blur', 'resize', 'sharpen', 'jpeg']:
        # for attack in ['median']:#['awgn', 'median', 'blur', 'resize', 'sharpen']:
        with open(f'{attack}-test.csv', 'w') as output_file:
            if attack == 'awgn':
                for std in [1, 2, 3, 5, 11, 13, 15, 17]:
                    alpha = 0.35
                    print(f"Starting {std}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = awgn(watermarked_im, std, 234565432)
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{std};{__wp};{output}")
                        output_file.write(f"{alpha};{std};{__wp};{output}\n")
            elif attack == 'jpeg':
                for qf in [90, 85, 80, 70, 50, 20, 10, 5]:
                    alpha = 0.35
                    print(f"Starting {qf}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = jpeg_compression(watermarked_im, qf)
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{qf};{__wp};{output}")
                        output_file.write(f"{alpha};{qf};{__wp};{output}\n")
            elif attack == 'median':
                for ker in [1, 3, 5, 7, 9]:
                    alpha = 0.35
                    print(f"Starting {ker}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = median(watermarked_im, ker)
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{ker};{__wp};{output}")
                        output_file.write(f"{alpha};{ker};{__wp};{output}\n")
            elif attack == 'blur':
                for blr in [1, 2, 4, 8, 10, 12, 15]:
                    alpha = 0.35
                    print(f"Starting {blr}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = blur(watermarked_im, [blr, blr])
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{blr};{__wp};{output}")
                        output_file.write(f"{alpha};{blr};{__wp};{output}\n")
            elif attack == 'resize':
                for res in [0.9, 0.7, 0.5, 0.4, 0.3, 0.2]:
                    alpha = 0.35
                    print(f"Starting {res}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = resizing(watermarked_im, res)
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{res};{__wp};{output}")
                        output_file.write(f"{alpha};{res};{__wp};{output}\n")
            elif attack == 'sharpen':
                for sig in [1, 1.33, 1.5, 1.66, 2]:
                    alpha = 0.35
                    print(f"Starting {sig}")
                    while alpha >= 0.25:
                        alpha -= 0.01
                        x, watermarked_im = embedding(ori_path, mark_path, alpha=alpha)
                        attacked_im = sharpening(watermarked_im, sig, 1)
                        watermark_original = extract_wm(ori_image, watermarked_im, alpha=alpha)
                        watermark_attacked = extract_wm(ori_image, attacked_im, alpha=alpha)
                        sim_wt = similarity(watermark_attacked, watermark_original)
                        output = 1 if sim_wt > THRESHOLD else 0
                        __wp = wpsnr(watermarked_im, attacked_im)
                        print(f"{alpha};{sig};{__wp};{output}")
                        output_file.write(f"{alpha};{sig};{__wp};{output}\n")
    output_file.close()


def test_boato_wrapper():
    names = [
        'buildings', 'tree', 'rollercoaster'
    ]
    groupname = 'nonsonoprompt'
    for name in names:
        test_boato(name, groupname)


def test_boato(im_name, groupname, ext='bmp'):
    original = '%s.%s' % (im_name, ext)
    watermarked = '%s_%s.%s' % (im_name, groupname, ext)

    # TIME REQUIREMENT: the detection should run in < 5 seconds
    start_time = time.time()
    tr, w = mod.detection(original, watermarked, watermarked)
    end_time = time.time()
    if (end_time - start_time) > 5:
        print('ERROR! Takes too much to run: ' + str(end_time - start_time))

    # THE WATERMARK MUST BE FOUND IN THE WATERMARKED IMAGE
    if tr == 0:
        print('ERROR! Watermark not found in watermarked image')

    # THE WATERMARK MUST NOT BE FOUND IN ORIGINAL
    tr, w = mod.detection(original, watermarked, original)
    if tr == 1:
        print('ERROR! Watermark found in original')

    # CHECK DESTROYED IMAGES
    img = cv2.imread(watermarked, 0)
    attacked = []
    c = 0
    ws = []

    attacked.append(blur(img, 15))
    attacked.append(awgn(img, 50, 123))
    attacked.append(resizing(img, 0.1))

    for i, a in enumerate(attacked):
        a_name = 'attacked-%d.bmp' % i
        cv2.imwrite(a_name, a)
        tr, w = mod.detection(original, watermarked, a_name)
        if tr == 1:
            c += 1
            ws.append(w)
        os.remove(a_name)
    if c > 0:
        print('ERROR! Watermark found in %d destroyed images with ws %s' % (c, str(ws)))

    # CHECK UNRELATED IMAGES
    files = [os.path.join('images/test', f) for f in os.listdir('images/test')]
    c = 0
    for f in files:
        tr, w = mod.detection(original, watermarked, f)
        if tr == 1:
            c += 1
    if c > 0:
        print('ERROR! Watermark found in %d unrelated images' % c)

    # CHECK MARK
    immm = cv2.imread(original, 0)
    watt = cv2.imread(watermarked, 0)
    x = np.load('nonsonoprompt.npy')
    x_star = mod.extract_wm(immm, watt)
    x_star = np.rint(abs(x_star)).astype(int)
    res = [1 for a, b in zip(x, x_star) if a == b]
    if sum(res) != 1024:
        print('The marks are different, please check your code')
    print(sum(res))


def calculate_threshold():
    sys.stderr = None
    images = [
        'buildings', 'rollercoaster', 'tree'
    ]

    mark_name = 'nonsonoprompt.npy'
    ext = 'bmp'

    scores = []
    labels = []

    for image_name in images:
        ori_path = '%s.%s' % (image_name, ext)
        scores, labels = generate_scores_labels(ori_path, mark_name, scores, labels)
        print(f"l(scores): {len(scores)}, l(labels): {len(labels)}")

    threshold = compute_threshold(scores, labels)

    print(f"T: {threshold}")


def alpha_calculation():
    images = [
        'buildings', 'rollercoaster', 'tree'
    ]

    image_name = images[2]

    mark_path = "nonsonoprompt.npy"
    group_name = 'nonsonoprompt'
    ext = 'bmp'
    ori_path = 'competition-day/%s.%s' % (image_name, ext)
    wat_path = '%s_%s.%s' % (image_name, group_name, ext)

    ori_image = cv2.imread(ori_path, 0)
    alpha = 0.45
    while alpha >= 0.10:
        alpha = alpha - 0.01
        # print(f"{ori_path, mark_name, alpha}")
        x, wat_image = embedding(ori_path, mark_path, alpha=alpha)
        cv2.imwrite(wat_path, wat_image)
        x_star = mod.extract_wm(ori_image, wat_image)
        sim_wt = mod.similarity(x, x_star)
        __wp = wpsnr(ori_image, wat_image)
        res = check_mark(x, x_star)
        print(f"{alpha};{__wp};{sim_wt};{1 if sim_wt < THRESHOLD else 0};{sum(res)}")


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    np.seterr(all='raise')

    print("Do Alpha-Attack-Test? [y]")
    choice = input()
    if choice == 'y':
        alpha_attack_test()

    print("Do Alpha-Calculation? [y]")
    choice = input()
    if choice == 'y':
        alpha_calculation()

    print("Compute threshold? [y]")
    choice = input()
    if choice == 'y':
        calculate_threshold()

    print("Do Boato-provided test? [y]")
    choice = input()
    if choice == 'y':
        test_boato_wrapper()

    print("Do Desperation-Test? [y]")
    choice = input()
    if choice == 'y':
        desperation_test()
