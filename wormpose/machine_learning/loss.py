"""
Definition of the loss function for the network
"""

import tensorflow as tf


def angle_diff(a, b):
    """
    Root Mean Square Error of the angle difference.
    The angle difference function takes into account the periodicity of angles
    """
    diff = tf.atan2(tf.sin(a - b), tf.cos(a - b))
    return tf.sqrt(tf.reduce_mean(tf.square(diff), axis=1))


def symmetric_angle_difference(y_true, y_pred):
    """
    We calculate the angle difference between the prediction and the two possible labels,
    and pick the minimum of the two,
    we average the result on the batch
    """

    dists = [angle_diff(y_pred, y_true[:, 0]), angle_diff(y_pred, y_true[:, 1])]
    mins = tf.reduce_min(dists, axis=0)
    loss = tf.reduce_mean(mins)
    return loss
