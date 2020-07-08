"""
Exports WormPose results (skeleton only) in WCON format

It requires the original WCON file from Zenodo as an input so that we can write the same metadata
"""
import json
import os
import zipfile

import h5py
import numpy as np

import wormpose
from wormpose import BaseResultsExporter
from wormpose.dataset.loaders.tierpsy.features_dataset import (
    get_frames_filename,
    get_stage_position,
    get_ratio_microns_pixels,
)


class ResultsExporter(BaseResultsExporter):
    """
    Exports WormPose results to WCON
    If there is an already existing WCON, load it to copy the metadata field
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def _find_wcon_file(self, video_name) -> str:
        """
        Will unzip the wcon file if necessary
        """
        wcon_filename = os.path.join(self.dataset_path, video_name, video_name + ".wcon")
        if os.path.exists(wcon_filename):
            return wcon_filename

        wcon_zipfilename = os.path.join(self.dataset_path, video_name, video_name + ".wcon.zip")
        if os.path.exists(wcon_zipfilename):
            with zipfile.ZipFile(wcon_zipfilename, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.dataset_path, video_name))
            return wcon_filename

        raise FileNotFoundError(f"Missing the original WCON file for {video_name}")

    def export(self, video_name: str, **kwargs):

        timestamp = kwargs["dataset"].features_dataset[video_name].timestamp
        results_skeletons: np.ndarray = kwargs["results_skeletons"]
        out_dir: str = kwargs["out_dir"]
        compress: bool = kwargs["compress"] if "compress" in kwargs else True

        try:
            wcon_filename = self._find_wcon_file(video_name)
            with open(wcon_filename, "r") as f:
                metadata = json.load(f)["metadata"]
        except Exception:
            metadata = {}
        wcon_content = {"metadata": metadata, "units": {}, "data": [{"id": "1"}]}

        h5file = get_frames_filename(self.dataset_path, video_name)
        with h5py.File(h5file, "r") as f:
            stage_position_pix = get_stage_position(f)
            ratio_microns_pixel = get_ratio_microns_pixels(f)
            frames_timestamp = f["timestamp"]["time"][:]

        wcon_content["metadata"]["software"] = {
            "name": "WormPose (https://github.com/iteal/wormpose)",
            "version": wormpose.__version__,
        }
        wcon_content["units"] = {"x": "micrometers", "y": "micrometers", "t": "seconds"}

        none_results = [None] * len(results_skeletons[0])

        data = wcon_content["data"][0]
        t_series = []
        x_series = []
        y_series = []

        for cur_time, skel in enumerate(results_skeletons):
            frame_index = np.where(timestamp == cur_time)[0]
            if len(frame_index) == 0:
                continue
            result_skel_microns = (skel + stage_position_pix[frame_index[0]]) / ratio_microns_pixel
            if np.any(np.isnan(result_skel_microns)):
                x_series.append(none_results)
                y_series.append(none_results)
            else:
                x_series.append(result_skel_microns[:, 0].tolist())
                y_series.append(result_skel_microns[:, 1].tolist())
            t_series.append(frames_timestamp[frame_index[0]])

        data["x"] = x_series
        data["y"] = y_series
        data["t"] = t_series

        exported_wcon = os.path.join(out_dir, video_name + ".wcon")
        with open(exported_wcon, "w") as f:
            json.dump(wcon_content, f)

        if compress:
            with zipfile.ZipFile(exported_wcon + ".zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(exported_wcon, arcname=os.path.basename(exported_wcon))

            os.remove(exported_wcon)
