import os

import cv2
import numpy as np

from neigh_search import LSH
from feat_extract import FeatureExtractor

labels = []
vectors = []

for image in os.listdir("./data"):
    filename = "./data/" + image
    img = cv2.imread(filename)
    labels.append(image.split(".")[0])
    vectors.append(np.resize(FeatureExtractor(img).get_features(), 100))

labels = np.array(labels)
vectors = np.array(vectors)
print(vectors.shape)

lsh = LSH(vectors, labels)
lsh.build()
lsh.write("index")

knns = lsh.query(vectors)
print(knns)
