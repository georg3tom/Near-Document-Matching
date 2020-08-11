import cv2
import numpy as np
import os

from neigh_search import Lsh
from feat_extraction import FeatureExtractor

if __name__ == '__main__':
    labels = []
    vectors = []

    for image in os.listdir('./data'):
        filename = './data/' + image
        img = cv2.imread(filename)
        labels.append(image.split('.')[0])
        vectors.append(np.resize(FeatureExtractor(img), 100))

    labels = np.array(labels)
    vectors = np.array(vectors)
    print(vectors.shape)

    lsh = Lsh(vectors, labels)
    lsh.build()

    knns = lsh.query(vectors)
    print(knns)
