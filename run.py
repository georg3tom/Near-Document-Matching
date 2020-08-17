import os

import cv2
import numpy as np
import time

from neigh_search import LSH
from feat_extract import FeatureExtractor

from helpers import Logger

labels = []
vectors = []
imgPath = "./data/"

l = Logger()

for image in os.listdir(imgPath):
    filename = imgPath + image
    l.log(f"reading {filename}")
    img = cv2.imread(filename)
    labels.append(image.split(".")[0])
    l.log("extracting features...")
    st = time.time()
    vectors.append(FeatureExtractor(img, window_stride={"h": 1, "w": 1}).get_features())
    en = time.time()
    l.log(f"done, took {en - st}")

labels = np.array(labels)
vectors = np.array(vectors)

lsh = LSH(vectors, labels)
lsh.build()
lsh.write("./index", "./labels")

knns = lsh.query(vectors)
print(knns)
