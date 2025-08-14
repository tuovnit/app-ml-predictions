''' This file contains the functions for performing cross-validation.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import itertools
from typing import List, Tuple

import pandas as pd

from data import load_data
from evaluate import accuracy
from model import init_perceptron, PERCEPTRON_VARIANTS
from sklearn.metrics import f1_score

def cross_validation(
        cv_folds: List[pd.DataFrame], 
        perceptron_variant: str, 
        lr_values: list, 
        mu_values: list,
        epochs: int = 10) -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of pandas DataFrames, corresponding to folds of the data. 
            The first column of each DataFrame, called "label", corresponds to y, 
            while the remaining columns are the features x.
        perceptron_variant (str): The variant of the perceptron algorithm to use
        lr_values (list): a list of learning rate hyperparameter values to try
        mu_values (list): a list of margin (mu) hyperparameter values to try
        epochs (int): how many epochs to train each model for. Defaults to 10

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hints:
        - We've provided a helper function `init_perceptron()` in model.py to initialize your model. 
          You can call it with `model = init_perceptron(perceptron_variant, num_features=num_features, lr=lr, mu=mu)`
        - The python `itertools.product()` function returns the Cartesian product of multiple lists.
          You can call `itertools.product(lr_values, mu_values) to get all combinations as (lr, mu) tuples.
        - You can convert a pandas DataFrame to a numpy ndarray with `df.to_numpy()`
    '''

    best_hyperparams = {'lr': None, 'mu': None}
    best_avg_f1_score = 0


    # YOUR CODE HERE
    num_features = cv_folds[0].shape[1] - 1
    
    for lr, mu in itertools.product(lr_values, mu_values):
        f1_scores = []
        
        for i in range(len(cv_folds)):
            train_folds = cv_folds[:i] + cv_folds[i+1:]
            train_data = pd.concat(train_folds)
            test_data = cv_folds[i]
            
            x_train = train_data.drop('label', axis=1).to_numpy()
            y_train = train_data['label'].to_numpy()
            x_test = test_data.drop('label', axis=1).to_numpy()
            y_test = test_data['label'].to_numpy()
            
            model = init_perceptron(perceptron_variant, num_features=num_features, lr=lr, mu=mu)
            model.train(x_train, y_train, epochs=epochs)
            y_pred = model.predict(x_test)
            
            f1_scores.append(f1_score(y_test, y_pred, average=None)[-1])
                    
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        if avg_f1_score > best_avg_f1_score:
            best_avg_f1_score = avg_f1_score
            best_hyperparams['lr'] = lr
            best_hyperparams['mu'] = mu

    return best_hyperparams, best_avg_f1_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-validation for different hyperparameters')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=PERCEPTRON_VARIANTS, 
        help=f'Which perceptron model to run. Must be one of {PERCEPTRON_VARIANTS}.')
    parser.add_argument('--lr_values', nargs='+', type=float, default=[1], 
        help='A list (space separated) of learning rate (eta) values to try. This is the same as the initial learning rate.')
    parser.add_argument('--mu_values', nargs='+', type=float, default=[0], 
        help='A list (space separated) of margin (mu) values to try. Defaults to [0].')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    cv_folds = data_dict['cv_folds']

    # run cross_validation
    print(f'run cross-validation')
    best_hyperparams, best_f1_score = cross_validation(
        cv_folds=cv_folds, 
        perceptron_variant=args.model,
        lr_values=args.lr_values, 
        mu_values=args.mu_values, 
        epochs=args.epochs)
    
    # print best hyperparameters and accuracy
    print('\nbest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\n       F1 score: {best_f1_score:.3f}\n')
