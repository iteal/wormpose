"""
Simple BaseFramePreprocesing implementation
"""

from typing import Optional, Tuple, Callable

import cv2
import numpy as np

from wormpose.dataset.base_dataset import BaseFramePreprocessing
from wormpose.dataset.image_processing.image_utils import segment_foreground, OtsuThreshold


class SimpleFramePreprocessing(BaseFramePreprocessing):
    def __init__(
        self,
        foreground_dilate_struct_element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        foreground_close_struct_element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        threshold_fn: Callable = OtsuThreshold(blur_kernel=(5, 5)),
    ):
        self.foreground_dilate_struct_element = foreground_dilate_struct_element
        self.foreground_close_struct_element = foreground_close_struct_element
        self.threshold_fn = threshold_fn

    def process(self, frame: np.ndarray, background_threshold: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return segment_foreground(
            frame, self.foreground_close_struct_element, self.foreground_dilate_struct_element, self.threshold_fn,
        )
