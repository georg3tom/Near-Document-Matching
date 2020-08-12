"""
Returns the edges detected in an image
"""

import cv2
import numpy as np


def hog(img):
    """
    Returns the canny edges detected in the image
    Expects an RGB image (3 channels)
    """

    return cv2.HOGDescriptor().compute(img)


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    print("Loaded image")

    h = hog(img)
    print("Calculation done")

    print(h.shape)
