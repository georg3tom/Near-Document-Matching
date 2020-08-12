"""
Histogram of Oriented Gradients
"""

import cv2
import numpy as np


def hog(img):
    """
    Returns the HOG descriptors for an image
    a HUGE return, have to use wisely
    Expects an RGB image (3 channels)
    """

    return cv2.HOGDescriptor().compute(img)


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    print("Loaded image")

    h = hog(img)
    print("Calculation done")

    print(h.shape)
