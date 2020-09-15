"""
Returns the surf descriptors of an image
"""

import cv2
import numpy as np


def surf(img):
    """
    Returns the surf descriptors of image 
    Expects an RGB image (3 channels)
    """

    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)

    return des


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = sift(img)
    print(h.shape)
