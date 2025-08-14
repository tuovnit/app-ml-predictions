''' This file contains the code for finding the optimal training epochs.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
from typing import Tuple

import numpy as np

from data import load_data
from evaluate import accuracy
from model import init_perceptron, Model, PERCEPTRON_VARIANTS
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

def train_epochs(
        model: Model, 
        train_x: np.ndarray, 
        train_y: np.ndarray, 
        val_x: np.ndarray, 
        val_y: np.ndarray,
        epochs: int = 20) -> Tuple[int, float]:
    '''
    Run epoch training to find the ideal number of epochs.

    Args:
        model (Model): the perceptron model to train
        train_x (np.ndarray): a numpy ndarray containing the training features
        train_y (np.ndarray): a numpy ndarray containing the training labels
        val_x (np.ndarray): a numpy ndarray containing the validation features
        val_y (np.ndarray): a numpy ndarray containing the validation labels
        epochs (int): the maximum number of epochs to train for. Defaults to 20

    Returns:
        int: the optimal number of epochs
        float: the validation accuracy corresponding to the optimal number of epochs
    '''
    
    best_epochs = -1
    best_val_f1_score = 0
    f1_scores = []
    total_updates = 0

    # YOUR CODE HERE
    for epoch in range(1, epochs + 1):
        model.train(train_x, train_y, epochs=1)
        f1 = f1_score(val_y, model.predict(val_x), average=None)[-1]
        f1_scores.append(f1)
        
        if f1 > best_val_f1_score:
            best_epochs = epoch
            best_val_f1_score = f1
            
        total_updates += model.update_count

    return best_epochs, best_val_f1_score



# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=PERCEPTRON_VARIANTS, 
        help=f'Which perceptron model to run. Must be one of {PERCEPTRON_VARIANTS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='How many epochs to train for. Defaults to 20.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    train_x = data_dict['train'].drop('label', axis=1).to_numpy()
    train_y = data_dict['train']['label'].to_numpy()
    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    val_x = data_dict['val'].drop('label', axis=1).to_numpy()
    val_y = data_dict['val']['label'].to_numpy()
    print(f'  val x shape: {val_x.shape}\n  val y shape: {val_y.shape}')

    # load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    model = init_perceptron(
        variant=args.model, 
        num_features=train_x.shape[1], 
        lr=args.lr, 
        mu=args.mu)
    print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

    # train the model to find optimal number of epochs
    best_epochs, best_f1_score = train_epochs(
        model=model, 
        train_x=train_x, 
        train_y=train_y, 
        val_x=val_x, 
        val_y=val_y,
        epochs=args.epochs)
    print(
        f'\noptimal number of epochs from epoch training' +
        f':\n\n         epochs: {best_epochs:5d}\n       F1 score: {best_f1_score:.3f}\n')
