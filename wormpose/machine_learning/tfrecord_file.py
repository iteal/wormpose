"""
Functions to read and write TFrecord files containing wormpose labeled data:
 one image paired with centerline angles (two options for head-tail flip)
"""

from functools import partial
from typing import Tuple, List

import tensorflow as tf

from wormpose.machine_learning.generic_file_writer import GenericFileWriter


def get_tfrecord_dataset(
    filenames, image_shape: Tuple[int, int], batch_size: int, theta_dims: int, is_train: bool,
):
    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP")

    if is_train:
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()

    dataset = dataset.map(
        partial(parse_example_normalize_image, theta_dims=theta_dims, image_shape=image_shape,),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def parse_example(example, theta_dims: int):
    features = {
        "data": tf.io.FixedLenFeature([], tf.string),
        "label0": tf.io.FixedLenFeature([theta_dims], tf.float32),
        "label1": tf.io.FixedLenFeature([theta_dims], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example, features)
    return parsed_features


def parse_example_normalize_image(example, theta_dims: int, image_shape: Tuple[int, int]):
    parsed_features = parse_example(example, theta_dims)

    img = tf.io.decode_raw(parsed_features["data"], tf.uint8)
    img = tf.reshape(img, (image_shape[0], image_shape[1]))
    img = img[:, :, tf.newaxis]
    img = tf.cast(img, tf.float32)
    img /= 255.0

    labels = [parsed_features["label0"], parsed_features["label1"]]

    return img, labels


class Writer(object):
    def __init__(self, filename: str):
        self.filename = filename
        self.record_writer = None

    def __enter__(self):
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        self.record_writer = tf.io.TFRecordWriter(self.filename, options=options)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.record_writer.close()

    def write(self, image_data, label_0, label_1):
        feature = {
            "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.tostring()])),
            "label0": tf.train.Feature(float_list=tf.train.FloatList(value=label_0.tolist())),
            "label1": tf.train.Feature(float_list=tf.train.FloatList(value=label_1.tolist())),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        self.record_writer.write(example_proto.SerializeToString())


def parse_square_image(example, theta_dims: int):
    parsed_features = parse_example(example, theta_dims)
    img = tf.io.decode_raw(parsed_features["data"], tf.uint8)
    img_size = tf.math.sqrt(tf.cast(tf.shape(img)[0], tf.float32))
    img = tf.reshape(img, (img_size, img_size))
    return img, [parsed_features["label0"], parsed_features["label1"]]


def read(filename: str, theta_dims: int):
    """
    Read a tfrecord file where the images in the files have the same width and height
    """
    raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(partial(parse_square_image, theta_dims=theta_dims))

    for parsed_record in parsed_dataset:
        yield parsed_record


def write_training_data_to_tfrecord(f, image_data, label_data, **kwargs):
    f.write(image_data, label_data[0], label_data[1])


class TfrecordLabeledDataWriter(GenericFileWriter):
    def __init__(self, filename: str):
        open_file = partial(Writer, filename)
        write_file = lambda f, data: write_training_data_to_tfrecord(f, **data)

        super().__init__(open_file=open_file, write_file=write_file)
