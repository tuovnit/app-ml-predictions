''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob
import pandas as pd
from sklearn.model_selection import KFold

def load_data() -> dict:
    '''
    Loads the data for one of the three datasets provided in this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cv data as (x, y) tuples of np.ndarray matrices
    '''

    # load train dataset
    train = pd.read_csv(f'../output/svm_logreg_train.csv')

    # load test dataset
    test = pd.read_csv(f'../output/svm_logreg_test.csv')

    # load cross validation datasets
    cv_folds = []
    # for cv_fold_path in glob(f'{dataset_path}/CVfolds/*'):
    #     fold = pd.read_csv(cv_fold_path)
    #     cv_folds.append(fold)
    
    kf = KFold(n_splits=5, shuffle=True)
    train_values = train.values

    for train_index, val_index in kf.split(train_values):
        fold = pd.DataFrame(train_values[val_index], columns=train.columns)
        cv_folds.append(fold)

    return {
        'train': train,
        'test': test,
        'cv_folds': cv_folds}
