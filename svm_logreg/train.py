''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from data import load_data
from evaluate import accuracy
from model import LogisticRegression, MajorityBaseline, Model, SupportVectorMachine, MODEL_OPTIONS
from sklearn.metrics import f1_score


def init_model(args: object, num_features: int) -> Model:
    '''
    Initialize the appropriate model from command-line arguments.

    Args:
        args (object): the argparse Namespace mapping arguments to their values.
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the hyperparameters in args.
    '''

    if args.model == 'majority_baseline':
        model = MajorityBaseline()
    
    elif args.model == 'svm':
        model = SupportVectorMachine(
            num_features=num_features, 
            lr0=args.lr0, 
            C=args.reg_tradeoff)

    elif args.model == 'logistic_regression':
        model = LogisticRegression(
            num_features=num_features, 
            lr0=args.lr0, 
            sigma2=args.reg_tradeoff)
    
    return model

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr0', type=float, default=0.1, 
        help='The initial learning rate hyperparameter gamma_0. Defaults to 0.1.')
    parser.add_argument('--reg_tradeoff', type=float, default=1, 
        help='The regularization tradeoff hyperparameter for SVM and Logistic Regression. Defaults to 1.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='How many epochs to train for. Defaults to 20.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    train_x = data_dict['train'].drop('label', axis=1).to_numpy()
    train_y = data_dict['train']['label'].to_numpy()
    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    test_x = data_dict['test'].drop('label', axis=1).to_numpy()
    test_y = data_dict['test']['label'].to_numpy()
    print(f'  test x shape: {test_x.shape}\n  test y shape: {test_y.shape}')

    # load the model
    print(f'initialize model')
    model = init_model(args=args, num_features=train_x.shape[1])
    print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

    # train the model
    if args.model == 'majority_baseline':
        print(f'train model')
        model.train(x=train_x, y=train_y)
    else:
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

    # evaluate model on train and test data
    print('evaluate')
    train_predictions = model.predict(x=train_x)
    train_accuracy = accuracy(labels=train_y, predictions=train_predictions)
    print(f'  train accuracy: {train_accuracy:.3f}')
    test_predictions = model.predict(x=test_x)
    test_accuracy = accuracy(labels=test_y, predictions=test_predictions)
    print(f'  test accuracy: {test_accuracy:.3f}')
    
    # Submission.
    eval = pd.read_csv('../output/svm_logreg_eval.csv')
    
    eval_x = eval.drop('label', axis=1)
    eval_y = eval['label'].to_list()
    
    eval_predictions = model.predict(x=eval_x)
    eval_accuracy = accuracy(labels=eval_y, predictions=eval_predictions)
    print(f'  eval accuracy: {eval_accuracy:.3f}')
    
    # Calculate F1 scores.
    train_f1 = f1_score(train_y, train_predictions, average=None)
    test_f1 = f1_score(test_y, test_predictions, average=None)
    eval_f1 = f1_score(eval_y, eval_predictions, average=None)
    
    print(f'  train F1 score: {train_f1[-1]:.3f}')
    print(f'  test F1 score: {test_f1[-1]:.3f}')
    print(f'  eval F1 score: {eval_f1[-1]:.3f}')
    
    data_frame = pd.DataFrame({
        'prediction': eval_predictions,
        'label': eval_y
    })
    
    data_frame.to_csv('svm_logreg_preds_no_id.csv', index=False)