import cv2
import pandas as pd
import shutil
import os
import ipdb

# Create a function to split data between train and validation and between 0 and 1 labels"


def subfolders(excel_file, img_folder, train_img, val_img):
    """
    Create a function to separate image data into subfolders
    input:
        excel_file: path to an excel file containing the filename, split, and class
        img_folder: path referring to the folder where the images are stored
        train_img: path to the folder where two subfolders (0 and 1) are initialized
        val_img: path to the folder where two subfolders (0 and 1) are initialized
    output:
         splits images into 4 subfolders: train0, train1, val0, and val1
    """
    labels_image = pd.read_csv(excel_file)

    # split train and val dataset
    train = labels_image.loc[labels_image["split"] == "train"]
    val = labels_image.loc[labels_image["split"] == "validation"]

    # split train and val between silo and non-silo
    train0 = train.loc[train["class"] == 0]
    train1 = train.loc[train["class"] == 1]
    val0 = val.loc[val["class"] == 0]
    val1 = val.loc[val["class"] == 1]

    # Move TRAIN images labeled 0 to the correct folder
    for i in train0.index:
        try:

            im = cv2.imread(os.path.join(img_folder, train0.loc[i, "filename"]))
            cv2.imwrite(os.path.join(train_img, "0", train0.loc[i, "filename"]), im)
        except:
            ipdb.set_trace()
    # Move TRAIN images labeled 1 to the correct folder
    for i in train1.index:
        im = cv2.imread(os.path.join(img_folder, train1.loc[i, "filename"]))
        cv2.imwrite(os.path.join(train_img, "1", train1.loc[i, "filename"]), im)

    for i in val0.index:
        im = cv2.imread(os.path.join(img_folder, val0.loc[i, "filename"]))
        cv2.imwrite(os.path.join(val_img, "0", val0.loc[i, "filename"]), im)

    # Move TRAIN images labeled 1 to the correct folder
    for i in val1.index:
        im = cv2.imread(os.path.join(img_folder, val1.loc[i, "filename"]))
        cv2.imwrite(os.path.join(val_img, "1", val.loc[i, "filename"]), im)

    # Remove hidden file
    try:
        shutil.rmtree("image/train/.ipynb_checkpoints")
        shutil.rmtree("image/val/.ipynb_checkpoints")
    except:
        pass


if __name__ == "__main__":
    subfolders
