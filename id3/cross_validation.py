''' This file contains the functions for performing cross-validation.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
from typing import Tuple

import pandas as pd

from data import load_data
from model import DecisionTree
from train import train, evaluate
from sklearn.metrics import f1_score


def cross_validation(cv_folds: list, depth_limit_values: list, ig_criterion: str = 'entropy') -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of dataframes, each corresponding to a fold of the data
        depth_limit_values (list): a list of depth_limit hyperparameter values to try
        ig_criterion (str): the information gain variant to use. Should be one of "entropy" or "collision".

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hint:
        - You'll need to instantiate a new DecisionTree object for every depth_limit value and fold
        - You should be able to reuse your train and evaluate functions from train.py
        - You can concatenate a list of dataframes with `pd.concat([df1, df2, ...], axis=1)`
    '''

    best_hyperparams = {'depth_limit': None}
    best_f1_score = 0


    # YOUR CODE HERE
    for depth_limit in depth_limit_values:
        f1_scores = []
        
        for i in range(len(cv_folds)):
            train_folds = cv_folds[:i] + cv_folds[i+1:]
            train_data = pd.concat(train_folds)
            test_data = cv_folds[i]
            
            model = DecisionTree(depth_limit, ig_criterion)
            train(model, train_data.drop('label', axis=1), train_data['label'])
            f1 = f1_score(test_data['label'], model.predict(test_data.drop('label', axis=1)), average=None)[-1]
            f1_scores.append(f1)
        
        average_f1_score = sum(f1_scores) / len(f1_scores)
        if average_f1_score > best_f1_score:
            best_f1_score = average_f1_score
            best_hyperparams['depth_limit'] = depth_limit
            
    return best_hyperparams, best_f1_score


# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--depth_limit_values', '-d', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], 
        help='The list (comma separated) of maximum depths to try.')
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use.')
    args = parser.parse_args()


    # load data
    data_dict = load_data()
    cv_folds = data_dict['cv_folds']

    # run cross_validation
    best_hyperparams, best_f1_score = cross_validation(
        cv_folds=cv_folds, 
        depth_limit_values=args.depth_limit_values, 
        ig_criterion=args.ig_criterion)
    
    # print best hyperparameters and accuracy
    print('\nBest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'       F1 score: {best_f1_score:.3f}\n')
