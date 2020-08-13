import os

import cv2
import numpy as np

from neigh_search import LSH
from feat_extract import FeatureExtractor

labels = []
vectors = []
imgPath = "./data_gen/output/"

for image in os.listdir(imgPath):
    filename = imgPath + image
    # print(filename)
    img = cv2.imread(filename)
    labels.append(image.split(".")[0])
    vectors.append(FeatureExtractor(img).get_features())

labels = np.array(labels)
vectors = np.array(vectors)

lsh = LSH(vectors, labels)
lsh.build()
lsh.write("./index","./labels")

knns = lsh.query(vectors)
print(knns)
