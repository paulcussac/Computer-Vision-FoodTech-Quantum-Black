# Quantum-Black-Challenge

## Data 

To properly get the data, one has to download it locally on his/her computer, unzip and put the folder in the main folder. One should have a folder like data/ai_ready/ with 6 different elements:
- images/ : folder containing the images
- masks/ : folder containing the masks
- train_images/ : folder containing two empty subfolders (0 and 1)
- val_images/ : folder containing two empty subfolders (0 and 1)
- test_images : folder containing one subfolder named 'test'
- x-ai_data.csv : csv file containing classes for each image.

## Repository description

### main folder
One can find in the main folder different python files used to train the models
- config.py : config file to precise all the parameters such as batch_size, learning_rate
- main.py : python folder to run to train the image classification model
- Image_Segmentation_Keras.ipynb : jupyter notebook to train and get the results of the image segmentation problem

Furthermore, main.py produces 3 different files :
- auc_curve.png : plot of the auc curve
- predictions.csv : predictions of our model stored in a proper csv file
- last_keras_model.h5 : file storing the weights of the image classification part

### model_building
Folder containing the python files to create the different models.

The most efficient model is stored in new_keras_model.py.

### model_test
Folder containing the python file to test the model and make prediction.

### app
Folder containing the file to create a web app that integrates the model and its predictions.

Webapp demo: [https://sdelgendre-webapp-foodix-home-7srx0f.streamlit.app/](https://sdelgendre-webapp-foodix-home-7srx0f.streamlit.app/)

**Instructions to reproduce the environment and run the app locally:**

- Create a new environment:

```pipenv shell```

- Install requirements:

```pip install -r requirements.txt```

- Launch the app:

```streamlit run Home.py```
