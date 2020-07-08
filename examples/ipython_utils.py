import shutil
import os
import tempfile
import cv2
import time

from IPython.display import Image, display, clear_output
from ipywidgets import interact


class ImagesViewer(object):
    def __init__(self, temp_dir=None):

        if temp_dir is None:
            temp_dir = tempfile.gettempdir()

        self.temp_dir = tempfile.mkdtemp(dir=temp_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.mkdir(self.temp_dir)

        self.count = 0
        self.filenames = []

    def add_image(self, image):
        filename = os.path.join(self.temp_dir, f"{self.count:09d}.png")
        self.filenames.append(filename)
        cv2.imwrite(filename, image)
        self.count += 1

    def add_image_filename(self, image_filename):
        self.filenames.append(image_filename)
        self.count += 1

    def view_as_slider(self):
        display_as_slider(self)

    def view_as_video(self, delay_secs):
        display_as_video(self, delay_secs=delay_secs)

    def view_as_list(self, legends):
        for filename, legend in zip(self.filenames, legends):
            print(legend)
            display(Image(open(filename, "rb").read()))


def display_as_slider(*img_viewers):
    def load_img(index=0):
        for img_viewer in img_viewers:
            display(Image(open(img_viewer.filenames[index], "rb").read()))

    interact(load_img, index=(0, len(img_viewers[0].filenames) - 1))


def display_as_video(*img_viewers, delay_secs):

    for index in range(len(img_viewers[0].filenames)):
        clear_output(wait=True)
        for img_viewer in img_viewers:
            display(Image(open(img_viewer.filenames[index], "rb").read()))
        time.sleep(delay_secs)
