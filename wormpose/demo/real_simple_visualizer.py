#!/usr/bin/env python

"""
Visualizer for the real processed images
"""

from wormpose.dataset.loader import load_dataset
from wormpose.dataset.loaders.resizer import add_resizing_arguments, ResizeOptions
from wormpose.images.real_dataset import RealDataset


class RealSimpleVisualizer(object):
    """
    Utility class to visualize the real images processed
    """

    def __init__(self, dataset_loader: str, dataset_path: str, video_name=None, **kwargs):
        resize_options = ResizeOptions(**kwargs)
        dataset = load_dataset(dataset_loader, dataset_path, resize_options=resize_options)

        self.video_name = video_name if video_name is not None else dataset.video_names[0]
        self.real_dataset = RealDataset(
            frame_preprocessing=dataset.frame_preprocessing, output_image_shape=dataset.image_shape,
        )
        self.frames_dataset = dataset.frames_dataset

    def generate(self):
        with self.frames_dataset.open(self.video_name) as frames:
            for frame in frames:
                processed_frame, _ = self.real_dataset.process_frame(frame)
                yield frame, processed_frame


def main():
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_loader", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument(
        "--video_name", type=str, help="Optional video name. If not set, only visualize one video.",
    )
    add_resizing_arguments(parser)
    args = parser.parse_args()

    real_visualizer = RealSimpleVisualizer(**vars(args))

    for orig, processed in real_visualizer.generate():
        cv2.imshow("orig", orig)
        cv2.imshow("processed", processed)
        cv2.waitKey()


if __name__ == "__main__":
    main()
