import os

import cv2
import numpy as np
from random import randint
import faiss

from neigh_search import LSH
from feat_extract import FeatureExtractor

vector = []
path = "./data_gen/output/"

images = os.listdir(path)
filename = path + images[randint(0, len(images))]
print(filename)
neigh = 10

img = cv2.imread(filename)
features = []
features.append(FeatureExtractor(img).get_features())
features = np.array(features)
features = features.astype("float32")
labels = np.fromfile("./labels", dtype="<U36")

index = faiss.read_index("./index")
distances, indices = index.search(features, neigh)
print(labels[np.array(indices)])
