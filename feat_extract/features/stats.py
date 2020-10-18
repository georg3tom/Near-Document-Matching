"""
Returns the common stats per channel for an image
"""

import cv2
import numpy as np


def stats(img):
    """
    Returns the common stats like mean, variance, etc per channel
    """
    # fail on grayscale images
    assert len(img.shape) >= 3

    # fail on non-rgb type images
    # assert img.shape[-1] == 3


    if len(img.shape) == 3:
        features = np.array([])

        for c in range(img.shape[-1]):
            img_slice = img[:, :, c].ravel()

            np.append(features, np.mean(img_slice))
            np.append(features, np.var(img_slice))
            np.append(features, np.median(img_slice))

        return features
    else:
        img = np.moveaxis(img, -1, -3)
        img = img.reshape((*img.shape[:-2], -1))
        features = np.append(np.mean(img, axis=-1), np.std(img, axis=-1), axis=-1)
        features = np.append(features, np.median(img, axis=-1), axis=-1)
        return features


if __name__ == "__main__":
    # img = cv2.imread("./test.png")
    # h = stats(img)

    # print(h)

    img = np.zeros((2,2,2,3))
    h = stats(img)
    print(h.shape)
