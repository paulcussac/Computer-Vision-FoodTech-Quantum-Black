
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore", UserWarning)


#Test functions
def test_model(test_g, model, batch_size):
    """
    Create the function to test the model and create predictions
    Input:
        test_g: test generator dataset
        model: saved model previously trained, in .h5 format
        batch_size: batch size for testing
    Output:
        NA"""

    test_g.reset()
    predictions = model.predict_generator(
        test_g,
        verbose=1,
        steps=200/batch_size
    )

    return predictions

def output_preds(preds, train_g, test_g, out):
    """
    Function to format the output with labels and predictions
    Input:
        preds: model predictions
        train_g: training generator to get the class labels
        test_g: training generator to get the image labels
        out: string, name of the csv to be exported
    Output:
        out: name of the csv file to be saved"""

    predicted_class_indices= np.round(preds).astype(int)
    labels = (train_g.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices[:,0]]

    filenames=test_g.filenames
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    results.to_csv(out+".csv", index=False)

    return results
