"""
Save results to a H5 file
"""

import os
import shutil
from typing import Optional

import h5py


class ResultsSaver(object):
    def __init__(self, temp_dir: str, results_root_dir: str, results_filename: str):
        self.temp_dir = temp_dir
        self.results_root_dir = results_root_dir
        self.results_filename = results_filename

    def save(self, results: dict, video_name: str, metadata: Optional[dict] = None):
        results_folder = os.path.join(self.results_root_dir, video_name)
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        results_temp_filepath = os.path.join(self.temp_dir, self.results_filename)
        with h5py.File(results_temp_filepath, "w") as f:

            if metadata is not None:
                for key, val in metadata.items():
                    f.attrs[key] = val

            for group_name, group_val in results.items():
                group = f.create_group(group_name)
                for name, val in group_val.__dict__.items():
                    if val is not None:
                        group.create_dataset(name, data=val)

        shutil.move(results_temp_filepath, os.path.join(results_folder, self.results_filename))
