"""
Returns the sift descriptors of an image
"""

import cv2
import numpy as np


def sift(img):
    """
    Returns the sift descriptors of image 
    Expects an RGB image (3 channels)
    """

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None) 

    return des


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = sift(img)
    print(h.shape)

