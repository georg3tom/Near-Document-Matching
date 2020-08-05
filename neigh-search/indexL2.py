"""
index data for k neighbour search
"""

import faiss
import numpy as np
import cv2

def IndexL2(data, d):
    """
    indexing the data
    """

    data = data.astype('float32')
    index = faiss.IndexFlatL2(d)
    # quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexIVFFlat(quantizer, d, data.shape[0])
    if not index.is_trained:
        index.train(data)
    index.add(data)
    return index

if __name__ == '__main__':
    d = 64
    nb = 10000
    nq = 1000
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # index = faiss.IndexFlatL2(d)
    # print(index.is_trained)
    # index.add(xb)
    # print(index.ntotal)

    k = 4
    # D, I = index.search(xb[:5], k)
    # print(I)
    # print(D)

    # D, I = index.search(xq, k)
    # print(I[:5])

    index = IndexL2(xb, d)
    _, I = index.search(xq, k)
    print(I)

