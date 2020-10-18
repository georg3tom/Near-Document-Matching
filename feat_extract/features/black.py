"""
Returns the no of black pixels
"""

import cv2
import numpy as np


def black(img, black_thresh=10):
    """
    Returns the number of black pixels
    """
    if len(img.shape) == 3:
        return np.array(np.count_nonzero(img <= black_thresh))
    else:
        feature = np.count_nonzero(img.reshape((*img.shape[:-3], -1)) <= black_thresh, axis=-1)
        return feature.reshape((*feature.shape, 1))


if __name__ == "__main__":
    # img = cv2.imread("./test.png")
    # h = black(img)
    # print(h)
    img = np.zeros((3,2,2,1))
    h = black(img)
    print(h)
