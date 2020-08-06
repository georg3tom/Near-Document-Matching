"""
Returns the no of black pixels
"""

import cv2
import numpy as np
from histogram import histogram


def black(img):
    """
    Returns the number of black pixels 
    Expects an RGB image (3 channels)
    """
    rang = 10
    hist = histogram(img)

    return int(np.sum(hist[:rang, :]))


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = black(img)
    print(h)

