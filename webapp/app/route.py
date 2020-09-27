import os
from os.path import realpath, dirname
import sys
import io
from PIL import Image

import cv2
import faiss
from flask import render_template, request, jsonify
import numpy as np
from app import app

sys.path.insert(
    0,
    realpath(os.path.join(dirname(realpath(__file__)), "../../")),
)

from neigh_search import LSH
from feat_extract import FeatureExtractor

# either use absolute path OR make sure that
# they are relative to `app`
labels_path = "../labels.npy"
index_path = "../index"

WEBAPP_CACHE = {}
WEBAPP_LABELS = np.load(labels_path)
WEBAPP_INDEX = faiss.read_index(index_path)


@app.route("/")
def hello_world():
    """
    Just for testing
    """
    return "Hello, World!"


@app.route("/query")
def query():
    """
    main page
    """
    return render_template("main.html")


@app.route("/search", methods=["POST"])
def search():
    """
    Hashes the image and searchs for nearest neighbors
    """
    fname = request.form["filename"]
    filepath = f"app/static/image/{fname}"
    if not os.path.isfile(filepath):
        print("Error, couldn't locate file")
        return jsonify(images={})

    img = cv2.imread(filepath)
    if img is None:
        print("Error, couldn't open file")
        return jsonify(images={})

    print(f"Opened file. Dimen: {img.shape}")

    # check cache
    print("Looking for image in cache")
    if fname in WEBAPP_CACHE:
        print("Found!")
        return WEBAPP_CACHE[fname]

    print("Couldn't find image in cache")
    print("Calculating features...")
    features = np.array([FeatureExtractor(img).get_features()], dtype=np.float32)
    print(f"Done!\nFeature shape: {features.shape}")

    print("Querying...")
    _, indices = WEBAPP_INDEX.search(features, 10)
    print("Done!")
    ret = jsonify(images=list(WEBAPP_LABELS[np.array(indices)][0]))
    WEBAPP_CACHE[fname] = ret

    return ret

@app.route("/searchUpload", methods=["POST"])
def searchUpload():
    """
    Hashes the image and searchs for nearest neighbors
    """
    print(request.files["file-0"])
    f = request.files["file-0"]

    img = Image.open(io.BytesIO(f.stream.read()))
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)

    if img is None:
        print("Error, couldn't open file")
        return jsonify(images={})

    print(f"Received file. Dimen: {img.shape}")

    print("Calculating features...")
    features = np.array([FeatureExtractor(img).get_features()], dtype=np.float32)
    print(f"Done!\nFeature shape: {features.shape}")

    print("Querying...")
    _, indices = WEBAPP_INDEX.search(features, 10)
    print("Done!")
    ret = jsonify(images=list(WEBAPP_LABELS[np.array(indices)][0]))

    return ret
