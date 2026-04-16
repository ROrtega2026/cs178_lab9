import json

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.manifold import Isomap

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def isomap(data, k=15, r=None):
    # TODO: replace with your isomap implementation
    return Isomap(n_neighbors=k, radius=r).fit_transform(data)


@app.route("/isomap", methods=["POST"])
def compute_isomap():
    req = request.get_json()
    data = np.array(req["data"])
    xy = None
    if "k" in req:
        xy = isomap(data, k=req["k"], r=None)
    elif "r" in req:
        xy = isomap(data, r=req["r"], k=None)
    return xy.tolist()


if __name__ == "__main__":
    app.run(port=5001)
