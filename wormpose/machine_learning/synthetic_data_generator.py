"""
Generates synthetic data using multiprocessing
"""

import os
import pickle
import time
from multiprocessing import Process, Value
from typing import Type, Callable, Generator, Optional

import numpy as np

from wormpose.dataset import Dataset
from wormpose.machine_learning.generic_file_writer import GenericFileWriter
from wormpose.pose.centerline import flip_theta


class _TemplatesChunkData(object):
    def __init__(self, num_total_templates):

        self.frames = None
        self.measurements = None
        self.skeletons = None
        self.video_names = None

        self._num_total = num_total_templates
        self._cur_index = 0

    def _init_chunk_arr(self, arr):
        arr = np.asarray(arr)
        return np.empty((self._num_total,) + arr.shape[1:], dtype=arr.dtype)

    def add_data(self, features, frames, template_indexes, video_name: str):

        # sort indexes for faster frames reading
        template_indexes = np.sort(template_indexes)

        if self._cur_index == 0:
            # init all chunk arrays
            # frames images may not have all the same size, so preallocate a simple python list instead
            self.frames = [None] * self._num_total
            self.skeletons = self._init_chunk_arr(features.skeletons)
            self.measurements = self._init_chunk_arr(features.measurements)
            self.video_names = self._init_chunk_arr(video_name)

        for i in range(len(template_indexes)):
            template_index = template_indexes[i]

            self.frames[self._cur_index] = frames[template_index]
            self.skeletons[self._cur_index] = features.skeletons[template_index]
            self.measurements[self._cur_index] = features.measurements[template_index]
            self.video_names[self._cur_index] = video_name

            self._cur_index = self._cur_index + 1

    def write_to_file(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def _write_to_file(
    out_filename: str,
    num_samples: int,
    template_filename: str,
    postures_generation_fn: Callable[[], Generator],
    progress_counter,
    writer: Type[GenericFileWriter],
    synthetic_dataset_args,
    random_seed: Optional[int],
):
    from wormpose.images.synthetic import SyntheticDataset

    seed = int.from_bytes(os.urandom(4), byteorder="little") if random_seed is None else random_seed
    np.random.seed(seed)

    with open(template_filename, "rb") as templates_f:
        templates = pickle.load(templates_f)
    os.remove(template_filename)

    worm_measurements = {}
    for video_name in np.unique(templates.video_names):
        indexes = np.where(templates.video_names == video_name)[0]
        measurements = templates.measurements[indexes]
        average_measurements = np.empty((measurements.shape[1:]), dtype=measurements.dtype)
        for name in measurements.dtype.names:
            average_measurements[0][name] = np.nanmean(measurements[name])
        worm_measurements[video_name] = average_measurements

    synthetic_dataset = SyntheticDataset(**synthetic_dataset_args)

    # preallocate all the template indexes to create the synthetic images (random)
    template_indexes = np.random.randint(len(templates.frames), size=num_samples)

    # preallocate all the choices for the position of the head (random)
    headtail_choice = np.random.choice([0, 1], size=num_samples)

    # preallocate image buffer to draw the synthetic image on
    image_data = np.empty(synthetic_dataset.output_image_shape, dtype=np.uint8)

    postures_gen = postures_generation_fn()
    with writer(out_filename) as synth_data_writer:

        for index, (cur_template_index, cur_headtail_choice) in enumerate(zip(template_indexes, headtail_choice)):
            theta = next(postures_gen)
            label_data = [theta, flip_theta(theta)]
            video_name = templates.video_names[cur_template_index]
            template_frame = templates.frames[cur_template_index]
            template_skeleton = templates.skeletons[cur_template_index]
            template_measurements = worm_measurements[video_name]
            synthetic_dataset.generate(
                theta=label_data[cur_headtail_choice],
                template_frame=template_frame,
                template_skeleton=template_skeleton,
                template_measurements=template_measurements,
                out_image=image_data,
            )
            synth_data_writer.write(locals())
            progress_counter.value = index + 1


class SyntheticDataGenerator(object):
    """
    This generator makes synthetic data by distributing the work to several processes:
    Each process will write one file containing the synthetic data and its label
    We use a "GenericFileWriter" to be able to write the files in different formats (TFrecord, pickle etc...)
    """

    _MAX_TEMPLATE_FRAMES = 1000

    def __init__(
        self,
        num_process: int,
        temp_dir: str,
        dataset: Dataset,
        postures_generation_fn: Callable[[], Generator],
        enable_random_augmentations: bool,
        writer: Type[GenericFileWriter],
        random_seed: Optional[int],
    ):
        """

        :param num_process: How many processes to distribute the work: there will be one file written per process
        :param temp_dir: Where to store temporary files
        :param dataset: A WormPose dataset
        :param postures_generation_fn: A function generating worm postures (angles)
        :param enable_random_augmentations: If true, the synthetic images will have random augmentations:
            translation, scale, blur
        :param writer: object responsible of writing the data to files
        :param random_seed: Seed for reproducibility
        """

        self.num_process = num_process
        self.temp_dir = temp_dir
        self.dataset = dataset
        self.postures_generation_fn = postures_generation_fn
        self.writer = writer
        self.random_seed = random_seed

        self.synthetic_dataset_args = dict(
            frame_preprocessing=dataset.frame_preprocessing,
            enable_random_augmentations=enable_random_augmentations,
            output_image_shape=dataset.image_shape,
        )

        self.video_names = dataset.video_names
        self._all_template_indexes = [dataset.features_dataset[v].labelled_indexes for v in self.video_names]

        if np.sum([len(x) for x in self._all_template_indexes]) == 0:
            raise RuntimeError("Can't create training data because couldn't find any labelled frame in the dataset.")

    def _split_data(self, num_samples_per_process):
        """
         Split a subset of template frames and associated features from different videos randomly to several processes
         """
        for process_index in range(self.num_process):

            num_samples = num_samples_per_process[process_index]
            if num_samples == 0:
                continue

            num_templates = min(num_samples, self._MAX_TEMPLATE_FRAMES)

            projected_num_templates_per_video = max(num_templates // len(self.video_names), 1)
            actual_num_templates_per_video = [
                min(projected_num_templates_per_video, len(x)) for x in self._all_template_indexes
            ]
            num_total_templates = np.sum(actual_num_templates_per_video)
            templates_chunk_data = _TemplatesChunkData(num_total_templates=num_total_templates)

            for video_name, cur_video_template_indexes, cur_video_num_templates in zip(
                self.video_names, self._all_template_indexes, actual_num_templates_per_video,
            ):
                # Select randomly some template indexes for each video
                template_indexes = np.random.choice(
                    cur_video_template_indexes, size=cur_video_num_templates, replace=False,
                )
                # Fill in the templates data chunk for this video
                with self.dataset.frames_dataset.open(video_name) as frames:
                    templates_chunk_data.add_data(
                        features=self.dataset.features_dataset[video_name],
                        frames=frames,
                        template_indexes=template_indexes,
                        video_name=video_name,
                    )

            template_filename = os.path.join(self.temp_dir, f"template_frames_{process_index:02d}.pkl")
            templates_chunk_data.write_to_file(template_filename)

            yield template_filename

    def generate(self, file_pattern: str, num_samples: int):
        """
        Start the generation of files containing synthetic data

        :param file_pattern: Filepath of the files to generate with an "index" variable
            ex: "path_to/file_{index}.tfrecord"
        :param num_samples: How many images (total) to generate, they will be split in several files
        """

        if num_samples <= 0:
            return 1  # nothing to do when num_samples is <= 0, progress 100%

        progress_counters = [Value("i", 0) for _ in range(self.num_process)]

        num_samples_per_process = [num_samples // self.num_process] * self.num_process
        num_samples_per_process[0] += num_samples % self.num_process

        workers = []
        for process_index, template_filename in enumerate(self._split_data(num_samples_per_process)):
            seed = self.random_seed + process_index if self.random_seed is not None else None
            out_filename = file_pattern.format(index=process_index)
            proc = Process(
                target=_write_to_file,
                args=(
                    out_filename,
                    num_samples_per_process[process_index],
                    template_filename,
                    self.postures_generation_fn,
                    progress_counters[process_index],
                    self.writer,
                    self.synthetic_dataset_args,
                    seed,
                ),
            )
            proc.start()
            workers.append(proc)

        while True:
            num_entries_written = np.sum([x.value for x in progress_counters])
            yield float(num_entries_written) / num_samples

            if num_entries_written >= num_samples:
                break
            time.sleep(0.5)

        for proc in workers:
            proc.join()
