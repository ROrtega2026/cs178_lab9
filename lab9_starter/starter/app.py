import json

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.manifold import Isomap
from scipy.sparse.csgraph import shortest_path
from sklearn.datasets import load_digits, make_s_curve, make_swiss_roll
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def isomap(data, k=15, r=None):
    # TODO: replace with your isomap implementation
    graph_adj_matrix = compute_graph(data, k, r)
    shortest_path_dist = shortest_path(graph_adj_matrix)
    return classical_mds(shortest_path_dist)


@app.route("/isomap", methods=["POST"])
def compute_isomap():
    req = request.get_json()
    dataset = req["data"]
    mode = req["mode"]
    xy = None
    labels = None

    if dataset == "swiss_roll":
        data, labels = make_swiss_roll(n_samples=1000)
    elif dataset == "s_curve":
        data, labels = make_s_curve(n_samples=1000)
    elif dataset == "digits":
        data = load_digits().data
        labels = load_digits().target

    if "k-nearest-neighbor" == mode:
        xy = isomap(data, k=int(float(req["param"])), r=None)
    elif "r-ball" == mode:
        xy = isomap(data, r=float(req["param"]), k=None)
    return {"data": xy.tolist(), "labels": labels.tolist()}

def euclidean_pairwise_distance(data):
    """
    Parameters
    ----------
    data : np.ndarray of shape (n, m)
        Data matrix containing n samples, each with m dimensions.

    Returns
    -------
    distances : np.ndarray of shape (n, n)
        Pairwise distance matrix, where entry (i, j) gives the distance
        between samples i and j.
    """
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances


def compute_graph(data, k=15, r=None):
    """
    Parameters
    ----------
    data : np.ndarray of shape (n, m)
        Data matrix containing n samples, each with m dimensions.

    Returns
    -------
    graph_adj_matrix : np.ndarray of shape (n, n)
        Graph adjacency matrix in which each nonzero entry (i, j) stores the
        distance between connected samples i and j. Entries with value 0 indicate
        that no edge exists between the corresponding samples.
    """
    n = data.shape[0]
    euclidean_pdist = euclidean_pairwise_distance(data)

    # Initialize compute adjacency matrix
    graph_adj_matrix = np.zeros([n, n])

    if r is not None:
        # r-ball neighborhood
        mask = (euclidean_pdist <= r) & (euclidean_pdist > 0)
        graph_adj_matrix[mask] = euclidean_pdist[mask]
    else:
        # k-nearest-neighbor graph
        k = min(k, n - 1)
        neighbor_idx = np.argsort(euclidean_pdist, axis=1)[:, 1:k + 1]

        rows = np.repeat(np.arange(n), k)
        cols = neighbor_idx.reshape(-1)
        vals = euclidean_pdist[rows, cols]

        graph_adj_matrix[rows, cols] = vals
        graph_adj_matrix[cols, rows] = vals

    return graph_adj_matrix

def classical_mds(pdist, n_components=2):
    n = pdist.shape[0]
    h = -np.ones([n, n]) / n + np.eye(n)
    sim = -1 / 2 * h @ pdist**2 @ h
    try:
        svd = np.linalg.svd(sim)
        xy = svd.U[:, :n_components] * svd.S[:n_components] ** 0.5
    except Exception as e:
        print(e)
        xy = np.random.randn(n,2)
    return xy

if __name__ == "__main__":
    app.run(port=5001)
