"""
Returns the histogram for an image for each channel present in it
"""

import cv2
import numpy as np


def histogram(img):
    """
    Returns the histogram (256, 3)
    Expects an RGB image (3 channels)
    """
    # fail on grayscale images
    assert len(img.shape) == 3

    hist_size = 256

    # upper bound excluded
    hist_range = (0, 256)

    n_channels = img.shape[-1]

    hist = cv2.calcHist(img, [0], None, [hist_size], hist_range, accumulate=False)

    for i in range(1, n_channels):
        hist = np.hstack(
            (
                hist,
                cv2.calcHist(img, [i], None, [hist_size], hist_range, accumulate=False),
            )
        )

    return hist


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = histogram(img)

    print(h.shape)
