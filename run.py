import os

import cv2
import numpy as np
import time

from neigh_search import LSHIndex as LSH, L2ExactIndex as L2
from feat_extract import FeatureExtractor

from helpers import Logger

labels = []
vectors = []
imgPath = "./data_gen/output/"

l = Logger()

images = [f for f in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath, f))]
total = len(images)
time_taken = 0

for i, image_name in enumerate(images):
    l.log("\033[2J\033[0;0H")
    l.log(f"Image {i+1} of {total}")

    filename = os.path.join(imgPath, image_name)

    l.log(f"reading {filename}")

    img = cv2.imread(filename, 0)
    img = np.reshape(img, (*img.shape, 1))

    labels.append(image_name)

    l.log("extracting features...")
    st = time.time()
    vectors.append(
        FeatureExtractor(
            img, preprocess_config={"do_fix_channels": False}
        ).get_features()
    )
    en = time.time()

    l.log(f"done, took {en - st:.2}s")
    time_taken += en - st
    l.log(f"Total: {time_taken:.2}s")

labels = np.array(labels)  # .astype(object)
vectors = np.array(vectors)

lsh = LSH(vectors, labels)
lsh.build(num_bits=32)
lsh.write("./index", "./labels")

dist, knns = lsh.query(vectors)
print("lsh")
print("distance:", dist[:5])
print("labels:", knns[:5])
print("accuracy:", lsh.score(vectors, labels))

l2 = L2(vectors, labels)
l2.build()

dist, knns = l2.query(vectors)
print("l2")
print("distance:", dist[:5])
print("labels:", knns[:5])
print("accuracy: ", l2.score(vectors, labels))
