import os

import cv2
import numpy as np
import time

from neigh_search import LSH, L2
from feat_extract import FeatureExtractor

from helpers import Logger

labels = []
vectors = []
imgPath = "./data/"

l = Logger()

images = os.listdir(imgPath)
total = len(images)
time_taken = 0

for i, image in enumerate(images):
    l.log("\033[2J\033[0;0H")
    l.log(f"Image {i+1} of {total}")
    filename = imgPath + image
    l.log(f"reading {filename}")
    img = cv2.imread(filename)
    labels.append(image.split(".")[0])
    l.log("extracting features...")
    st = time.time()
    vectors.append(FeatureExtractor(img, window_size={"h": 25, "w": 25}, window_stride={"h": 25, "w": 25}).get_features())
    en = time.time()
    l.log(f"done, took {en - st:.2}s")
    time_taken += en - st
    l.log(f"Total: {time_taken:.2}s")

labels = np.array(labels)
vectors = np.array(vectors)

lsh = LSH(vectors, labels)
lsh.build()
lsh.write("./index", "./labels")

dist, knns = lsh.query(vectors)
print('lsh')
print('distance:', dist)
print('labels:', knns)
print('accuracy:', lsh.score(vectors, labels))

l2 = L2(vectors, labels)
l2.build()

dist, knns = l2.query(vectors)
print('l2')
print('distance:', dist)
print('labels:', knns)
print('accuracy: ', l2.score(vectors, labels))
