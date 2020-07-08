"""
Generates evaluation data: random real processed images with labels and save them to a Tfrecord file
"""

import csv
import logging
import os

import numpy as np

from wormpose.dataset import Dataset
from wormpose.images.real_dataset import RealDataset
from wormpose.machine_learning import tfrecord_file
from wormpose.pose.centerline import skeleton_to_angle, flip_theta

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate(dataset: Dataset, num_samples: int, theta_dims: int, file_pattern: str,) -> int:
    """
    Generates evaluation dataset composed of processed real images, saved to a .TFrecord

    :param dataset: WormPose Dataset
    :param num_samples: How many images to generate
    :param theta_dims: Dimensions of theta for the labels
    :param file_pattern: Path of the output files with "index" variable
        example: "path_to_out/eval_{index}.tfrecord"
    :return: How many samples where actually generated (if there is less data than the requested num_samples)
    """
    labelled_frames = {}
    for video_name in dataset.video_names:
        skel_is_not_nan = ~np.any(np.isnan(dataset.features_dataset[video_name].skeletons), axis=(1, 2))
        labelled_indexes = np.where(skel_is_not_nan)[0]
        if len(labelled_indexes) > 0:
            labelled_frames[video_name] = labelled_indexes

    if len(labelled_frames) == 0:
        raise RuntimeError("Can't create evaluation data because couldn't find any labelled frame in the dataset.")

    len_labelled_frames = int(np.sum([len(x) for x in labelled_frames.values()]))
    if len_labelled_frames < num_samples:
        logging.warning(
            f"Not enough labelled frames in the dataset "
            f"to create an evaluation set of {num_samples} unique samples, "
            f"using all available {len_labelled_frames} samples instead."
        )
        num_samples = len_labelled_frames

    real_dataset = RealDataset(dataset.frame_preprocessing, dataset.image_shape)

    tfrecord_filename = file_pattern.format(index=0)
    csv_infos_filename = os.path.splitext(tfrecord_filename)[0] + ".csv"

    # get num_samples total random labelled frames from the videos
    eval_frames = _populate_eval_frames(labelled_frames, num_samples)

    # write the eval.tfrecord file with the images and the labels, save also the source infos in a separate eval.csv
    # the frames are not shuffled by video, all the frames from one video are consecutive in the file
    with tfrecord_file.Writer(tfrecord_filename) as record_writer, open(csv_infos_filename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for video_name, cur_video_eval_indexes in eval_frames.items():

            with dataset.frames_dataset.open(video_name) as frames:

                for eval_frame_index in cur_video_eval_indexes:
                    image_data, _ = real_dataset.process_frame(frames[eval_frame_index])
                    cur_skel = dataset.features_dataset[video_name].skeletons[eval_frame_index]
                    cur_theta = skeleton_to_angle(cur_skel, theta_dims=theta_dims)
                    cur_theta_flipped = flip_theta(cur_theta)

                    record_writer.write(image_data, cur_theta, cur_theta_flipped)
                    csv_writer.writerow([video_name, int(eval_frame_index)])

    return num_samples


def _populate_eval_frames(labelled_frames: dict, num_samples: int):
    cur_index = 0
    eval_frames = {}

    while cur_index < num_samples:
        # pick randomly a video name and a valid frame index for that video, without repetition
        chosen_video_name = np.random.choice(list(labelled_frames.keys()))
        valid_index_in_video = labelled_frames[chosen_video_name]
        pick_index_at = np.random.randint(0, len(valid_index_in_video))
        chosen_frame_index = valid_index_in_video[pick_index_at]
        # remove chosen frame from the available frames
        labelled_frames[chosen_video_name] = np.delete(labelled_frames[chosen_video_name], pick_index_at)
        if len(labelled_frames[chosen_video_name]) == 0:
            del labelled_frames[chosen_video_name]

        if chosen_video_name in eval_frames:
            eval_frames[chosen_video_name].append(chosen_frame_index)
        else:
            eval_frames[chosen_video_name] = [chosen_frame_index]
        cur_index += 1

    # sort for faster read access
    for video_name in eval_frames:
        eval_frames[video_name].sort()

    return eval_frames
