"""
Implementation of BaseFeaturesDataset to load Tierpsy tracker features
"""

import glob
import json
import os

import h5py
import numpy as np
from xml.dom import minidom

from wormpose.dataset.base_dataset import BaseFeaturesDataset


def get_frame_rate(f):
    mask_attrs = f["mask"].attrs
    if "fps" in mask_attrs:
        frame_rate = mask_attrs["fps"]
    elif "expected_fps" in mask_attrs:
        frame_rate = mask_attrs["expected_fps"]
    else:
        frame_rate = 1
    return frame_rate


def get_ventral_side(f):
    # Trying to get the ventral side attribute (optional)
    try:
        ventral_side = json.loads(f["experiment_info"][()].decode("utf-8"))["ventral_side"]
    except:
        ventral_side = None
    return ventral_side


def get_stage_position(f):
    len_frames = len(f["mask"])
    try:
        stage_position_pix = f["stage_position_pix"][:]
    except:
        stage_position_pix = np.zeros((len_frames, 2), dtype=float)
    return stage_position_pix


def get_ratio_microns_pixels(f):
    try:
        xml_info = f["xml_info"][()]
        xml = minidom.parseString(xml_info)
        microns = xml.getElementsByTagName("microns")[0].getElementsByTagName("x")[0].firstChild.nodeValue
        pixels = xml.getElementsByTagName("pixels")[0].getElementsByTagName("x")[0].firstChild.nodeValue
        ratio_microns_pixel = abs(float(microns)) / abs(float(pixels))
    except:
        ratio_microns_pixel = 1.0

    return ratio_microns_pixel


def _match_indexes(x, y):
    sorted_index = np.searchsorted(x, y)
    yindex = np.take(np.arange(len(x)), sorted_index, mode="clip")
    mask = x[yindex] != y
    result = np.ma.array(yindex, mask=mask)
    return result


def _get_width_measurement(name, ratio_microns_pixel, features_timeseries):
    feature_names = features_timeseries.dtype.names
    measurement_name = list(filter(lambda x: name in x and "width" in x, feature_names))[0]
    measurement = features_timeseries[measurement_name] * ratio_microns_pixel
    return measurement


def get_skeletons_timestamp(features_f, skeletons, features_timestamp):
    if "trajectories_data" in features_f:

        trajectories_data = features_f.get("trajectories_data")
        dt = [("skeleton_id", int)]
        with trajectories_data.astype(dt):
            skeletons_id = trajectories_data["skeleton_id"][:]

        skeletons_timestamp = np.zeros(len(skeletons), dtype=int)
        for i in range(len(skeletons)):
            skeletons_timestamp[i] = np.where(skeletons_id == i)[0][0]
    else:
        skeletons_timestamp = features_timestamp

    return skeletons_timestamp


def _resample(series, cur_timestamp, new_timestamp):
    len_new = len(new_timestamp)
    resampled_series = np.full((len_new,) + series.shape[1:], np.nan, dtype=series.dtype)

    matched_indexes = _match_indexes(cur_timestamp, new_timestamp)

    for index in range(len_new):

        if not matched_indexes.mask[index]:
            resampled_series[index] = series[matched_indexes[index]]

    return resampled_series


def get_features_filename(root_dir: str, name: str):
    """
    The features filename has different formats:
    ex: videoname_features.hdf5 or sometimes videoname_featuresN.hdf5
    """
    return glob.glob(os.path.join(root_dir, name, name + "*features*.hdf5"))[0]


def get_frames_filename(root_dir: str, name: str):
    return os.path.join(root_dir, name, name + ".hdf5")


def _read_features(root_dir, name):
    h5file = get_frames_filename(root_dir, name)
    h5featurefile = get_features_filename(root_dir, name)

    with h5py.File(h5file, "r") as f:
        frames_timestamp = f["timestamp"]["raw"][:]
        frame_rate = get_frame_rate(f)
        ventral_side = get_ventral_side(f)
        stage_position_pix = get_stage_position(f)
        ratio_microns_pixel = get_ratio_microns_pixels(f)

    with h5py.File(h5featurefile, "r") as f:
        skeletons = f["coordinates"]["skeletons"][:]
        features_timeseries = get_features_timeseries(f)
        features_timestamp = features_timeseries["timestamp"].astype(int)
        skeletons_timestamp = get_skeletons_timestamp(f, skeletons, features_timestamp)

        head_width = _get_width_measurement("head", ratio_microns_pixel, features_timeseries)
        midbody_width = _get_width_measurement("midbody", ratio_microns_pixel, features_timeseries)
        tail_width = _get_width_measurement("tail", ratio_microns_pixel, features_timeseries)
        measurements = np.stack([head_width, midbody_width, tail_width], axis=1)

    measurements = _resample(measurements, cur_timestamp=features_timestamp, new_timestamp=frames_timestamp)
    skeletons = _resample(skeletons, cur_timestamp=skeletons_timestamp, new_timestamp=frames_timestamp)

    # convert skeletons coordinates from microns to pixels
    skeletons = skeletons * ratio_microns_pixel - stage_position_pix[:, np.newaxis, :]

    head_width = measurements[:, 0]
    midbody_width = measurements[:, 1]
    tail_width = measurements[:, 2]

    return {
        "skeletons": skeletons,
        "head_width": head_width,
        "midbody_width": midbody_width,
        "tail_width": tail_width,
        "frame_rate": frame_rate,
        "ventral_side": ventral_side,
        "timestamp": frames_timestamp,
    }


def get_features_timeseries(f):
    features_timeseries = f[list(filter(lambda x: "timeseries" in x, f))[0]][:]
    # only use one worm (smallest index)
    worm_indexes = features_timeseries["worm_index"]
    worm_index = np.min(worm_indexes)
    cur_worm = np.where(worm_indexes == worm_index)[0]
    features_timeseries = features_timeseries[cur_worm]
    return features_timeseries


class FeaturesDataset(BaseFeaturesDataset):
    def __init__(self, dataset_path, video_names):
        self._features = {}
        for video_name in video_names:
            self._features[video_name] = _read_features(dataset_path, video_name)

    def get_features(self, video_name):
        return self._features[video_name]
