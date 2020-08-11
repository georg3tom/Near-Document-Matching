"""
https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6
"""

import faiss
import numpy as np
import cv2

class ExactIndex():
	def __init__(self, vectors, labels):
		self.dimension = vectors.shape[1]
		self.vectors = vectors.astype('float32')
		self.labels = labels    
   
	def build(self):
		self.index = faiss.IndexFlatL2(self.dimension,)
		self.index.add(self.vectors)
		
	def query(self, vectors, k=10):
		distances, indices = self.index.search(vectors, k) 
		# I expect only query on one vector thus the slice
		return [self.labels[i] for i in indices[0]]


if __name__ == '__main__':
	d = 64
	nb = 100
	nq = 10
	np.random.seed(1234)
	labels = np.array([i for i in range(nb)])
	xb = np.random.random((nb, d)).astype('float32')
	xb[:, 0] += np.arange(nb) / 1000.
	xq = np.random.random((nq, d)).astype('float32')
	xq[:, 0] += np.arange(nq) / 1000.

	index = ExactIndex(xb, labels)
	index.build()

	print(index.query(xq[:1]))