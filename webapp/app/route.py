from flask import render_template, url_for, request, jsonify
from app import app

import os
import cv2
import numpy as np
import faiss

from app.feat_extract import FeatureExtractor

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/query')
def query():
    return render_template('main.html')

@app.route('/search', methods=['POST'])
def search():
    fname = request.form['filename']
    try:
        query = []
        imgPath = './app/static/image/'
        if fname not in os.listdir(imgPath):
            return jsonify({'images': []})
        filename = imgPath + fname
        img = cv2.imread(filename)
        feat = FeatureExtractor(img, window_size={"h": 25, "w": 25}, window_stride={"h": 25, "w": 25}).get_features()
        query.append(feat)

        index = faiss.read_index('./app/static/index')
        labels = np.load('./app/static/labels.npy')
        k = min(6, labels.shape[0])
        query = np.array(query).astype("float32")
        _, indices = index.search(query, k)
        images = list(labels[np.array(indices)[0]])
        for i in range(k):
            images[i] = str(images[i]) + '.png'
        return jsonify({'images': images})

    except:
        print('error')
        return jsonify({'images': []})