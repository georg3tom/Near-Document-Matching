"""
Returns the no of black pixels
"""

import cv2
import numpy as np


def black(img, black_thresh=10):
    """
    Returns the number of black pixels
    """
    return np.array(np.count_nonzero(img <= black_thresh))


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = black(img)
    print(h)
