import cv2
import numpy as np
from skimage.filters import gaussian


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def color_change(norm_image, parsing, part=[11, 12], color=[230, 50, 20]):
    image = (norm_image * 255.).astype(np.uint8)

    b, g, r = color
    tar_color = np.zeros_like(image, dtype=np.uint8)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part[0] == 13:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
    elif part[0] == 11 or part[1] == 12:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    changed = sharpen(changed)

    if part[0] == 13:
        changed[parsing != part[0]] = image[parsing != part[0]]
        return changed
    else:
        area1 = parsing != part[0]
        area2 = parsing != part[1]
        area_test1 = area1 & area2
        changed[area_test1] = image[area_test1]
        return changed
