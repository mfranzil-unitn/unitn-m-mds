import random

import cv2

from nonsonoprompt.attacks.attacks import awgn, blur, jpeg_compression, median, resizing, sharpening

images = [
    'buildings', 'rollercoaster', 'tree'
]

attacks = [
    awgn, blur, jpeg_compression, median, resizing, sharpening
]

groups = [
    "duebrioches",
    "eagleview",
    "hideseekers",
    "mario",
    "mastersofentropy",
    "nonsonoprompt",
    "panebbianco",
    "paradisepark",
    "pythonized",
    "thelenasboys",
    "unitrentomag",
    "watermarkcharmer"
]


def attacker():
    loop = '_'
    while loop != 'e':
        print("Choose the image to attack:")
        for index, path in zip(range(len(images)), images):
            print(f"{index}: {path}; ", end="")

        im_index: int = int(input())

        print("G:", end='')
        for group, att in zip(range(len(groups)), groups):
            print(f"{group}: {att}; ", end='')

        group_index: int = int(input())
        group_name = str(groups[group_index])

        wat_path = f"{group_name}_{images[im_index]}.bmp"
        ori_path = f"{images[im_index]}.bmp"
        att_path = "nonsonoprompt_" + wat_path

        ori_image = cv2.imread(ori_path, 0)
        wat_image = cv2.imread(wat_path, 0)

        loop = '_'
        att_image = None
        while loop != 'e' and loop != 'n':
            print("A:", end='')
            for index, att in zip(range(len(attacks)), attacks):
                print(f"{index}: {att.__name__}; ", end='')

            att_index: int = int(input())

            if att_image is not None:
                wat_image = att_image

            if att_index == 0:
                print("====> std = ____ [], seed = ____ [-1 for random]")
                std = float(input())
                seed = int(input())
                if seed == -1:
                    seed = random.randint(0, pow(2, 16))

                attacked = awgn(wat_image, std, seed)

            elif att_index == 1:
                print("====> size of blur: x = ____ [], y = ____ []")
                x = int(input())
                y = int(input())

                attacked = blur(wat_image, [x, y])

            elif att_index == 2:
                print("====> QF = ____ [99, 98, 97, ..., 75, ...]")
                qf = int(input())

                attacked = jpeg_compression(wat_image, qf)

            elif att_index == 3:
                print("====> size of sliding window: x = ____ [], y = ____ []")
                x = int(input())
                y = int(input())

                attacked = median(wat_image, [x, y])

            elif att_index == 4:
                print("====> scale = ____ [0.9 ... 0.1]")
                scale = float(input())
                attacked = resizing(wat_image, scale)

            elif att_index == 5:
                print("====> sigma = ____ [max 1/2], alpha = ____ []")
                sigma = float(input())
                alpha = float(input())
                attacked = sharpening(wat_image, sigma, alpha)

            else:
                attacked = wat_image

            cv2.imwrite(att_path, attacked)
            result, wpsnr = detectors[group_name](ori_path, wat_path, att_path)
            att_image = attacked.copy()
            print(f"R: {result}, wpsnr: {wpsnr}")

            print("Type e to exit, n to start with a fresh image, anything to continue")
            loop = input()


if __name__ == '__main__':
    attacker()
