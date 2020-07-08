"""
This module assembles image data batches by preprocessing each image with a BaseFramePreprocessing object
It distributes the work to several processes and provides each batch as they are ready.
"""

import math
import os
import pickle
from multiprocessing import Process, Manager

import numpy as np

from wormpose import BaseFramePreprocessing
from wormpose.dataset import Dataset
from wormpose.images.real_dataset import RealDataset


class _ImagesBatch(object):
    def __init__(self, batch_size: int, image_shape):
        self.data = np.zeros(([batch_size, image_shape[0], image_shape[1], 1]), dtype=np.float32)
        self.index = 0

    def reset(self):
        self.index = 0

    def add(self, image: np.ndarray):
        self.data[self.index, :, :, 0] = image
        self.index += 1


def _assemble_images_batch(
    frame_preprocessing: BaseFramePreprocessing,
    data_reading_queue,
    results_queue,
    temp_dir: str,
    batch_size: int,
    image_shape,
):

    images_batch = _ImagesBatch(batch_size, image_shape)
    real_dataset = RealDataset(frame_preprocessing, image_shape)
    while True:
        queue_data = data_reading_queue.get()
        if queue_data is None:
            break

        data_filename, chunk_index = queue_data

        with open(data_filename, "rb") as f:
            images_batch.reset()
            while True:
                try:
                    raw_frame = pickle.load(f)
                except EOFError:
                    break
                cur_frame, _ = real_dataset.process_frame(raw_frame)
                images_batch.add(cur_frame)

        os.remove(data_filename)

        # normalize between  and 1
        images_data_batch = images_batch.data / 255.0

        image_filename = os.path.join(temp_dir, f"real_topredict_{chunk_index:09d}.npy")
        np.save(image_filename, images_data_batch)
        results_queue.put(image_filename)


class PredictDataGenerator(object):
    def __init__(
        self, dataset: Dataset, num_process: int, temp_dir: str, image_shape, batch_size: int,
    ):
        self.dataset = dataset
        self.num_process = num_process
        self.temp_dir = temp_dir
        self.image_shape = image_shape
        self.batch_size = batch_size

    def _read_data(self, data_reading_queue, num_frames: int, video_name: str):

        with self.dataset.frames_dataset.open(video_name) as frames:
            num_chunks = math.ceil(float(num_frames) / self.batch_size)
            for chunk_index in range(num_chunks):
                start = chunk_index * self.batch_size
                end = min(num_frames, start + self.batch_size)

                data_filename = os.path.join(self.temp_dir, f"data_to_predict_{chunk_index}.npy")
                with open(data_filename, "wb") as f:
                    for index in range(start, end):
                        pickle.dump(frames[index], f)

                data_reading_queue.put((data_filename, chunk_index))

        for _ in range(self.num_process):
            data_reading_queue.put(None)

    def run(self, video_name: str):

        manager = Manager()
        data_reading_queue = manager.Queue()
        results_queue = manager.Queue()

        workers = [
            Process(
                target=_assemble_images_batch,
                args=(
                    self.dataset.frame_preprocessing,
                    data_reading_queue,
                    results_queue,
                    self.temp_dir,
                    self.batch_size,
                    self.image_shape,
                ),
            )
            for _ in range(self.num_process)
        ]
        for w in workers:
            w.start()

        self._read_data(
            data_reading_queue=data_reading_queue,
            num_frames=self.dataset.num_frames(video_name),
            video_name=video_name,
        )

        for w in workers:
            w.join()

        results_files = []
        while not results_queue.empty():
            results_files.append(results_queue.get())
        results_files = sorted(results_files)

        for f in results_files:
            batch_data = np.load(f)
            yield batch_data
            os.remove(f)
