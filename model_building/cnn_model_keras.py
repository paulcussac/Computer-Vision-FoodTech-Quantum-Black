import config
import numpy as np
import pandas as pd

import keras
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Activation, LeakyReLU, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
import config


def train_set(train_path, image_size, batch_size):
    """
    Create train dataset to train the model
    Input:
        train_path: path referring to where the train images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        train_ds: train set in Keras format
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.05,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds


def val_set(val_path, image_size, batch_size):
    """
    Create val dataset to train the model
    Input:
        val_path: path referring to where the val images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        val_ds: train set in Keras format
    """
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        validation_split=0.05,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )

    return val_ds


def make_model(input_shape: tuple, num_classes: int):
    """
    function building a keras deep learning model with data augmentation for image recognition
    args:
        - input_shape : tuple containing the dimensions of the pictures. e.g input_shape = (180,180) when our input image is of shape 180*180
        - num_classes : int, number of different classes
    returns:
        - cnn_model : keras model with data augmentation and
    """
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    data_augmentation = create_data_augmentation_model()
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = Conv2D(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout)(x)
    x = LeakyReLU(alpha=0.2)(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 128]:
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = Dropout(config.dropout)(x)
        x = LeakyReLU(0.2)(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = layers.Dropout(config.dropout)(x)
        x = MaxPooling2D(3, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    x = Dense(100, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    cnn_model = keras.Model(inputs, outputs)
    return cnn_model


def create_data_augmentation_model():
    """
    Function to create the keras data_augmentation layer
    Input:
        - none
    Output:
        - data_augementation : keras layer, used in the make_model function
    """
    data_augmentation = keras.Sequential(
        [
            layers.Rescaling(1.0 / 255.0),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomContrast([0, 1]),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ]
    )
    return data_augmentation


def simple_model(num_classes: int):

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(256, 256, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model


def train_model(model, train_ds, val_ds, epochs):
    """
    Train a Keras CNN Model and output accuracy
    Input:
        model: keras cnn_model
        train_ds: training Keras Dataset
        val_ds: validation Keras dataset
        epochs: number of epochs to train the model
    Output:
        accuracy: accuracy measure of the classification
    """

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        # steps_per_epoch=1400//config.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
        # validation_steps = 1400//config.batch_size,
        verbose=2,
    )
