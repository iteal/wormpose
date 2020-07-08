#!/usr/bin/env python

import argparse
import logging
import tempfile

import h5py
import numpy as np
import numpy.ma as ma
import scipy.io
from sklearn.mixture import GaussianMixture

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _uniform_samples(a3s, n_bins=100, total_n_samples=20000):
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


def calculate_aic(eigenworms_matrix_path, shapes_file, coiled_modes_file, num_modes):
    eigenworms_matrix = np.loadtxt(eigenworms_matrix_path, delimiter=",").astype(np.float32)

    # Load angle library
    f = scipy.io.loadmat(shapes_file)
    thetas_w = ma.array(f["theta_ensemble"])
    thetas_w[thetas_w == 0] = ma.masked
    thetas_library_raw = ma.compress_rows(ma.vstack(thetas_w))
    raw_samples = thetas_library_raw[::2]

    # Load coiled modes library
    with h5py.File(coiled_modes_file, "r") as mat:
        refs = list(mat["#refs#"].keys())[1:]
        tseries_w = [ma.masked_invalid(np.array(mat["#refs#"][ref]).T)[:, :num_modes] for ref in refs]

    modes_library = ma.compress_rows(ma.vstack(tseries_w))
    # find indices with larger curvature (on the tail of the distribution of angles that can be solved)
    indices_curved = np.abs(modes_library[:, 2]) > np.percentile(raw_samples.dot(eigenworms_matrix[:, 2]), 95)
    curved_samples = modes_library[indices_curved].dot(eigenworms_matrix[:, :num_modes].T)

    # combine samples
    thetas_library_combined = np.vstack((curved_samples, raw_samples))

    # sample uniformly from various degrees of curvature
    indices = _uniform_samples(thetas_library_combined.dot(eigenworms_matrix[:, 2]))

    training_data = thetas_library_combined[indices]

    aic = []
    n_components_range = np.arange(150, 350, 10)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        try:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(training_data)
            aic.append(gmm.aic(training_data))
        except:
            aic.append(np.nan)

    return np.vstack((n_components_range, aic)).T


if __name__ == "__main__":
    import os
    import shutil
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes_file", type=str, help="Mat file with non coiled shapes", required=True)
    parser.add_argument("--coiled_modes_file", type=str, help="Mat file with coiled shapes (modes)", required=True)
    parser.add_argument("--eigenworms_matrix_path", type=str, required=True)
    parser.add_argument("--num_modes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out_file", type=str, default="aic_GMM.h5")
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp()

    for index in range(args.iterations):
        logger.info(f"Iteration {index}/{args.iterations}")
        results = calculate_aic(
            eigenworms_matrix_path=args.eigenworms_matrix_path,
            shapes_file=args.shapes_file,
            coiled_modes_file=args.coiled_modes_file,
            num_modes=args.num_modes,
        )
        np.savetxt(os.path.join(temp_dir, f"aic_{index}.txt"), results)

    all_files = list(sorted(glob.glob(os.path.join(temp_dir, "*.txt"))))
    all_res = [np.loadtxt(x) for x in all_files]
    shutil.rmtree(temp_dir)
    with h5py.File(args.out_file, "w") as f:
        f.create_dataset("aic_all", data=all_res)
