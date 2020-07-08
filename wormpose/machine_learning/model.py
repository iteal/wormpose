"""
Definition of the network model, a ResNet with three blocks
"""

import tensorflow as tf


def build_model(input_shape, out_dim, activation=tf.keras.layers.LeakyReLU):
    filters = 32

    def middle_stack(x, activation):
        x = _res_block(x, filters=filters, num_blocks=3, strides=2, name="res32", activation=activation,)
        x = _res_block(x, filters=filters * 2, num_blocks=3, strides=2, name="res64", activation=activation,)
        x = _res_block(x, filters=filters * 4, num_blocks=3, strides=2, name="res128", activation=activation,)
        return x

    return _build_model(input_shape, out_dim, middle_stack, filters, activation)


def _build_model(input_shape, out_dim, middle_stack, filters, activation):
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], 1))

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=7, strides=2, padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(x)

    x = middle_stack(x, activation=activation)

    x = tf.keras.layers.BatchNormalization()(x)
    x = activation()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=out_dim, name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def _basic_block(x, filters, strides, name, activation):
    x = tf.keras.layers.BatchNormalization(name=name + "_bn_0")(x)
    x = activation(name=name + "_act_0")(x)
    shortcut = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, padding="valid", strides=strides, name=name + "_conv_0",
    )(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides, name=name + "_conv_1",)(
        x
    )
    x = tf.keras.layers.BatchNormalization(name=name + "_bn_1")(x)
    x = activation(name=name + "_act_1")(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", strides=1, name=name + "_conv_2")(x)

    x = tf.keras.layers.Add(name=name + "_add")([x, shortcut])

    return x


def _res_block(x, filters, num_blocks, strides, name, activation):
    strides = [strides] + [1] * (num_blocks - 1)
    for index, stride in enumerate(strides):
        x = _basic_block(x=x, filters=filters, strides=stride, name=f"{name}_block_{index}", activation=activation,)
    return x
