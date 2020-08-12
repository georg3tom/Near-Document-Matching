"""
Returns the common stats per channel for an image
"""

import cv2
import numpy as np


def stats(img):
    """
    Returns the common stats like mean, variance, etc per channel
    Expects an RGB image (3 channels)
    """
    # fail on grayscale images
    assert len(img.shape) == 3

    # fail on non-rgb type images
    assert img.shape[-1] == 3

    features = []

    for c in range(img.shape[-1]):
        img_slice = img[:, :, c].ravel()

        features.append(np.mean(img_slice))
        features.append(np.var(img_slice))
        features.append(np.median(img_slice))

    return features


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    h = stats(img)

    print(h)
