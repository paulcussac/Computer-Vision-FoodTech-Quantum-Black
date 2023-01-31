import config
import numpy as np
import pandas as pd
import sys
import keras
from keras.initializers import RandomNormal
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    LeakyReLU,
    BatchNormalization,
)

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
import config
from model_test.test_model import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import warnings
import ipdb
import math
import cv2

warnings.simplefilter("ignore", UserWarning)


def myFunc(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # COLOR_RGB2HSV)


def train_set(train_path, image_size, batch_size):
    """
    Create train dataset to train the model
    Input:
        train_path: path referring to where the train images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        train_generator: train set in Keras format
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening =True,
        # # # brightness_range = [0.5, 2.0],
        # preprocessing_function = myFunc,
    )

    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=image_size, batch_size=batch_size, class_mode="binary"
    )
    print("train set generator created, test")
    return train_generator


def val_set(val_path, image_size, batch_size):
    """
    Create val dataset to train the model
    Input:
        val_path: path referring to where the val images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        val_generator: train set in Keras format
    """
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # preprocessing_function=myFunc,
    )

    val_generator = test_datagen.flow_from_directory(
        val_path, target_size=image_size, batch_size=batch_size, class_mode="binary"
    )

    return val_generator


def test_set(test_path, image_size, batch_size):
    """
    Create test dataset to test the model
    Input:
        test_path: path referring to where the test images are stored
        image_size: size of the images
        batch_size: batch size
    Output:
        test_generator: test set in Keras format
    """
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # preprocessing_function=myFunc,
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    return test_generator


def keras_model(input_shape):
    """
    function building a keras deep learning model for image recognition
    Inputs:
        - input_shape : tuple containing the dimensions of the pictures
        - train_g: training generator
        - val_g: validation generator
        - batch_size: batch size used for training
        - epochs: number of epochs for training
        - model_name: name to save the model
    Output:
        - keras_model : trained keras model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(LeakyReLU(0.1))
    # model.add(Dropout(config.dropout))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(0.1))
    # model.add(Dropout(config.dropout))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(0.1))
    # model.add(Dropout(config.dropout))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(LeakyReLU(0.2))
    # model.add(Dropout(config.dropout))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(config.dropout))
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(config.dropout))
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
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("starting training of the model")

    model.fit_generator(
        train_ds,
        steps_per_epoch=1400 // config.batch_size,
        epochs=config.number_epochs,
        validation_data=val_ds,
        validation_steps=400 // config.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    model.save("model_with_test.h5")

    preds = model.predict_generator(val_ds)
    number_of_examples = len(val_ds.filenames)
    number_of_generator_calls = math.ceil(
        number_of_examples / (1.0 * config.batch_size)
    )
    # 1.0 above is to skip integer division

    true_labels = []

    for i in range(0, int(number_of_generator_calls)):
        # print(np.array(val_ds[i][1]).shape[0])
        true_labels.extend(np.array(val_ds[i][1]))
    f1_score_val = f1_score(true_labels, 1 * (preds > 0.5))
    return f1_score_val
