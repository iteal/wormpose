#!/usr/bin/env python

import argparse
import gzip
import json
import logging
import time

import h5py
import numpy as np
import numpy.ma as ma
import scipy.io
from sklearn.mixture import GaussianMixture

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def uniform_samples(a3s, n_bins=100, total_n_samples=20000):
    samples_per_bin = int(total_n_samples / n_bins)
    step = 60 / n_bins
    indices = []
    for x0 in np.linspace(-30, 30, n_bins):
        xf = x0 + step
        sel = np.logical_and(a3s >= x0, a3s <= xf)
        if np.sum(sel) > 0:
            if np.sum(sel) < samples_per_bin:
                indices.append(np.arange(len(a3s))[sel])
            else:
                indices.append(np.random.choice(np.arange(len(a3s))[sel], samples_per_bin, replace=False))
    return np.hstack(indices)


def generate(shapes_file, coiled_modes_file, eigenworms_matrix_path, out_file, num_gaussians):
    # Load angle library from Greg
    f = scipy.io.loadmat(shapes_file)
    thetas_w = ma.array(f["theta_ensemble"])
    thetas_w[thetas_w == 0] = ma.masked
    thetas_library_raw = ma.compress_rows(ma.vstack(thetas_w))

    # Load library from Onno
    mat = h5py.File(coiled_modes_file, "r")
    refs = list(mat["#refs#"].keys())[1:]
    tseries_w = [ma.masked_invalid(np.array(mat["#refs#"][ref]).T)[:, :5] for ref in refs]
    mat.close()
    modes_library = ma.compress_rows(ma.vstack(tseries_w))

    eigenworms_matrix = np.loadtxt(eigenworms_matrix_path, delimiter=",").astype(np.float32)

    # same number of samples from full theta
    # raw_samples = thetas_library_raw[np.random.choice(np.arange(len(thetas_library_raw)),np.sum(indices_curved),replace=False)]
    raw_samples = thetas_library_raw[::2]

    # find indices with larger curvature
    indices_curved = np.abs(modes_library[:, 2]) > np.percentile(raw_samples.dot(eigenworms_matrix[:, 2]), 97.5)
    # get same number of samples from raw angles and projected modes
    curved_samples = modes_library[indices_curved].dot(eigenworms_matrix[:, :5].T)

    thetas_library_combined = np.vstack((curved_samples, raw_samples))

    indices = uniform_samples(thetas_library_combined.dot(eigenworms_matrix[:, 2]))
    training_data = thetas_library_combined[indices]

    # fit gaussian mixture model
    gmm = GaussianMixture(n_components=num_gaussians)
    gmm.fit(training_data)

    # sort according to curvature
    sorting_indices = np.argsort(np.sum(np.abs(np.diff(gmm.means_, axis=1)), axis=1))
    means = gmm.means_[sorting_indices]
    covariances = gmm.covariances_[sorting_indices]
    weights = gmm.weights_[sorting_indices]

    with gzip.open(out_file, "wt") as f:
        json.dump({"means": means.tolist(), "covariances": covariances.tolist(), "weights": weights.tolist()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes_file", type=str, help="Mat file with non coiled shapes")
    parser.add_argument("--coiled_modes_file", type=str, help="Mat file with coiled shapes (modes)")
    parser.add_argument("--eigenworms_matrix_path", type=str)
    parser.add_argument("--num_gaussians", type=int, default=270)
    parser.add_argument("--out_file", type=str, default="shapes_model.json.gz")
    args = parser.parse_args()

    start = time.time()
    generate(
        shapes_file=args.shapes_file,
        coiled_modes_file=args.coiled_modes_file,
        num_gaussians=args.num_gaussians,
        out_file=args.out_file,
        eigenworms_matrix_path=args.eigenworms_matrix_path,
    )
    end = time.time()
    logger.info(f"Finished setup of shapes generator model in {end - start:.1f}s with settings {vars(args)}")
