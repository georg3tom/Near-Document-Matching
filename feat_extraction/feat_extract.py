"""
gets an image, returns all extracted features
"""

import numpy as np
from .features import features


def getBlocks(img, num):
    x = img.shape[0] // num
    y = img.shape[1] // num
    blocks = []

    for i in range(0, num - 1):
        for j in range(0, num - 1):
            blocks.append(img[i * x : (i + 1) * x, j * y : (j + 1) * y, :])

        blocks.append(img[i * x : (i + 1) * x, (num - 1) * y :, :])

    for j in range(0, num - 1):
        blocks.append(img[(num - 1) * x :, j * y : (j + 1) * y, :])

    blocks.append(img[(num - 1) * x :, (num - 1) * y :, :])

    return blocks


def FeatureExtractor(img, num=2):
    blocks = getBlocks(img, num)
    feature_vector = np.zeros((0))

    for block in blocks:
        for feature in features:
            feature_vector = np.hstack([feature_vector, feature(block).ravel()])

    return feature_vector


# class FeatureExtractor:
#     pass
