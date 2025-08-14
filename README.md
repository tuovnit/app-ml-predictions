# Final Project Details

Note that the explore.ipynb file requires certain cells to be run before training with a certain model.
For each specific algorithm, training will give a predictions csv in its respective subfolder.
You then have to run a certain cell in the explore.ipybn file to convert that CSV into a valid submission csv.

The preprocessing in explore.ipynb will save the new data to the /output folder.
The postprocessing of the predictions in the respective algorithm's folder will save the final predictions to the /output folder.

The overall instructions on how to run the project can be followed through the explore.ipynb file, pay attention to the NOTEs commented.
The file outline below gives more detail on the file structure of the project.

# File Outline

Project | The overall folder of the project.
    adaboost    | The subfolder that contains the AdaBoost implementation.
        README.md   | Instructions on how to run AdaBoost code.
        train.py    | Contains the implementation of AdaBoost.
    id3 | The subfolder that contains the ID3 implementation.
        cross_validation.py | Contains code for cross validation of ID3.
        data.py | Contains utility code for loading the data for ID3.
        model.py    | Contains code for the ID3 model.
        README.md   | Instructions on how to run ID3 code.
        train.py    | Contains code for training and evaluating the ID3 model.
    output  | The subfolder that should contain all the saved preprocess data and final predictions from the algorithms.
    perceptron  | The subfolder that contains the Perceptron implementation.
        cross_validation.py | Contains the code for cross validation of Perceptron.
        data.py | Contains utility code for loading the data for Perceptron.
        epochs.py   | Contains code for finding optimal training epochs for Perceptron.
        evaluate.py | Contains code for evaluating model predictions for Perceptron.
        model.py    | Contains code for the Perceptron model.
        README.md   | Instructions on how to run Perceptron code.
        train.py    | Contains code for training and evaluating the Perceptron model.
    project_data    | The subfolder that contains all the project data.
        data    | The subfolder that contains: train.csv, test.csv, eval.anon.csv, and eval.id.
    svm_logreg  | The subfolder that contains the SVM and Logistic Regression implementations.
        cross_validation.py | Contains code for cross validation of SVM and Logistic Regression.
        data.py | Contains utility code for loading the data for SVM and Logistic Regression.
        evaluate.py | Contains code for evaluating model predictions for SVM and Logistic Regression.
        model.py    | Contains code for the SVM and Logistic Regression models.
        README.md   | Instructions on how to run SVM and Logistic Regression code.
        train.py    | Contains code for training and evaluating the SVM and Logistic Regression models.
        utils.py    | Contains code for utility functions for SVM and Logistic Regression.
    explore.ipynb   | Jupyter Notebook that contains initial exploration of the data, preprocessing for all the algorithms, and post processing for the final predictions.
    README.md   | Contains notes about the project and the file outline of the project.
    requirements.txt    | Libraries used in the project.