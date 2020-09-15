from flask import render_template, url_for, request, jsonify
from app import app

import os

import cv2
import numpy as np

from app.neigh_search import LSH, L2
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
        labels = []
        vectors = []
        query = []
        imgPath = './app/static/image/'
        images = os.listdir(imgPath)
        if fname not in images:
            return jsonify({'images': []})
        total = len(images)
        for i, image in enumerate(images):
            filename = imgPath + image
            img = cv2.imread(filename)
            labels.append(image)
            feat = FeatureExtractor(img, window_size={"h": 25, "w": 25}, window_stride={"h": 25, "w": 25}).get_features()
            vectors.append(feat)
            if image == fname:
                query.append(feat)

        labels = np.array(labels)
        vectors = np.array(vectors)
        query = np.array(query)

        lsh = LSH(vectors, labels)
        lsh.build()
        dist, knns = lsh.query(query)

        # l2 = L2(vectors, labels)
        # l2.build()
        # dist, knns = l2.query(vectors)
        return jsonify({'images': list(knns[0])})

    except:
        print('error')
        return jsonify({'images': []})