"""
https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6
"""

import faiss
import numpy as np
import cv2


class LSHIndex:
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype("float32")
        self.labels = labels
        self.index = None

    def build(self, num_bits=8):
        self.index = faiss.IndexLSH(self.dimension, num_bits)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        k = int(min(k, self.labels.shape[0]))
        vectors = vectors.astype("float32")
        distances, indices = self.index.search(vectors, k)
        return self.labels[np.array(indices)]

    def write(self, indexName, labelName):
        faiss.write_index(self.index, indexName)
        self.labels.tofile(labelName)


if __name__ == "__main__":
    d = 64
    nb = 100
    nq = 10
    np.random.seed(1234)
    labels = np.array(range(nb))
    xb = np.random.random((nb, d)).astype("float32")
    xb[:, 0] += np.arange(nb) / 1000.0
    xq = np.random.random((nq, d)).astype("float32")
    xq[:, 0] += np.arange(nq) / 1000.0

    index = LSHIndex(xb, labels)
    index.build()

    print(index.query(xq[:1]))
