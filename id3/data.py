''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd
from sklearn.model_selection import KFold

def load_data() -> dict:
    '''
    Loads all the data required for this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cross validation data as pandas dataframes
    '''

    # load train dataset
    #train = pd.read_csv('data/train.csv')
    train = pd.read_csv('../output/id3_train_discretized.csv')

    # load test dataset
    #test = pd.read_csv('data/test.csv')
    test = pd.read_csv('../output/id3_test_discretized.csv')

    # load cross validation datasets
    cv_folds = []
    # for cv_fold_path in glob('data/cv/*'):
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
