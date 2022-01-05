import cv2
import numpy as np

from nonsonoprompt.detection_nonsonoprompt import extract_wm, similarity
from nonsonoprompt.embedding_nonsonoprompt import embedding
from nonsonoprompt.wpsnr import wpsnr


def check_mark(X, X_star):
    X_star = np.rint(abs(X_star)).astype(int)
    res = [1 for a, b in zip(X, X_star) if a == b]
    return res


def tester(alpha):
    np.seterr(all='raise')
    images = [
        'buildings', 'rollercoaster', 'tree'
    ]

    # for i in range(1, 10):
    #     images.append(f'images/shared/000{i}')
    # for i in range(10, 30):
    #     images.append(f'images/shared/00{i}')
    # image_name = 'images/lena/lena'
    mark_name = 'nonsonoprompt.npy'
    group_name = 'nonsonoprompt'
    ext = 'bmp'

    np.seterr('raise')

    for image_name in images:
        ori_path = '%s.%s' % (image_name, ext)
        wat_path = '%s_%s.%s' % (image_name, group_name, ext)

        x, wat_image = embedding(ori_path,
                                 mark_name,
                                 alpha=alpha
                                 )

        cv2.imwrite(wat_path, wat_image)
        ori_image = cv2.imread(ori_path, 0)
        wat_image = cv2.imread(wat_path, 0)
        x_star = extract_wm(ori_image, wat_image,
                            alpha=alpha
                            )
        sim_wt = similarity(x, x_star)
        __wp = wpsnr(ori_image, wat_image)
        res = check_mark(x, x_star)
        count = sum(res)
        try:
            ori_image = None
            wat_image = None
            # os.remove(wat_path)
        except Exception as ex:
            pass
        print(f"{ori_path};{alpha};{__wp};{sim_wt};{count}")  # 1 if sim_wt < THRESHOLD else 0};{count}")


if __name__ == '__main__':
    tester(0.3)
