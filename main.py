## Load Packages
import numpy as np
import pandas as pd
import cv2
import os
import random

import sys
import config

sys.path.append("model_building/create_image_folders.py")
from model_building.create_image_folders import *

sys.path.append("model_building/keras_model.py")
# from model_building.cnn_model_keras import *
from model_building.new_keras_model import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import keras
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from tensorflow.keras import layers
import warnings

warnings.simplefilter("ignore", UserWarning)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


## Set paths
img_folder = os.path.join(os.getcwd(), "data", "ai_ready", "images")
train_img = os.path.join(os.getcwd(), "data", "ai_ready", "train_images")
val_img = os.path.join(os.getcwd(), "data", "ai_ready", "val_images")
test_img = os.path.join(os.getcwd(), "data", "ai_ready", "test_images")
labels_image = os.path.join(os.getcwd(), "data", "ai_ready", "x-ai_data.csv")
create_images = False

plot_auc = False


if __name__ == "__main__":
    tf.config.list_physical_devices()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ## Move images to subfolders
    if create_images:
        subfolders(labels_image, img_folder, train_img, val_img)

    df_labels = pd.read_csv(labels_image)
    ## Model
    # ipdb.set_trace()
    model = keras_model(input_shape=config.image_size + (3,))
    ## Train and Val dataset
    train_ds = train_set(train_img, config.image_size, config.batch_size)
    val_ds = val_set(val_img, config.image_size, config.batch_size)
    # test_ds = test_set(val_img, config.image_size, config.batch_size)
    ## Train Model
    f1_score_val = train_model(model, train_ds, val_ds, config.number_epochs)
    print(f"the f1_socre for the val set is :{f1_score_val}")
    test_ds = test_set(test_img, config.image_size, config.batch_size)

    y_preds=test_model(test_ds, model, 1)
    y_test = df_labels[df_labels['split']=="test"]['class'].values

    try:
        f1_score_test = f1_score(y_test, 1 * (y_preds > 0.5))
        accuracy_test = accuracy_score(y_test, 1 * (y_preds > 0.5))
        print(f"the f1_score test is {f1_score_test}, the accucary is {accuracy_test}")
        fpr, tpr, threshold = roc_curve(y_test, y_preds)
        roc_auc = auc(fpr, tpr)
    except:
        ipdb.set_trace()
    df_preds = df_labels[df_labels["split"] == "test"].copy()
    df_preds["preds_proba"] = y_preds
    df_preds.to_csv("class_predicted.csv", index=False)
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, color="#285430", label=f"AUC = {roc_auc :0.2f}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], color="#fed049", linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('auc_curve')
    plt.show()    
    output_preds(y_preds, train_ds, test_ds, 'class predicted2')

