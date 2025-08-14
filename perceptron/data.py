''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def load_data() -> dict:
    '''
    Loads all the data required for this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cv data as (x, y) tuples of np.ndarray matrices
    '''

    # load train dataset
    train = pd.read_csv('../output/perceptron_train.csv')
    # train = pd.read_csv('../output/perceptron_train_smote.csv')

    # load validation dataset
    val = pd.read_csv('../output/perceptron_eval.csv')
    # val = pd.read_csv('../output/perceptron_eval_smote.csv')

    # load test dataset
    test = pd.read_csv('../output/perceptron_test.csv')
    # test = pd.read_csv('../output/perceptron_test_smote.csv')

    # load cross validation datasets
    cv_folds = []
    
    # for cv_fold_path in glob('data/cv/*'):
    #     fold = pd.read_csv(cv_fold_path)
    #     cv_folds.append(fold)
    # last_index = 0    
    # split_index = int(len(train) / 5)
    # for i in range(1, 4):
    #     temp_fold = train.iloc[split_index * last_index:split_index * i, :]
    #     cv_folds.append(temp_fold)
    # cv_folds.append(train.iloc[split_index * 4:, :])
    
    # Create 5-fold cross-validation splits
    kf = KFold(n_splits=5, shuffle=True)
    train_values = train.values

    for train_index, val_index in kf.split(train_values):
        fold = pd.DataFrame(train_values[val_index], columns=train.columns)
        cv_folds.append(fold)

    return {
        'train': train,
        'val': val,
        'test': test,
        'cv_folds': cv_folds}
