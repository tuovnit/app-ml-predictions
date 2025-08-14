import argparse
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--lr', type=float, default=1,
                        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--nest', type=int, default=50,
                        help='The max number of estimators at which boosting is terminated. Defaults to 50.')
    args = parser.parse_args()
    
    # Load data.
    # train_data = pd.read_csv('../output/id3_train_discretized.csv')
    # test_data = pd.read_csv('../output/id3_test_discretized.csv')
    # eval_data = pd.read_csv('../output/id3_eval_discretized.csv')
    train_data = pd.read_csv('../project_data/data/train.csv')
    test_data = pd.read_csv('../project_data/data/test.csv')
    eval_data = pd.read_csv('../project_data/data/eval.anon.csv')
    
    train_x = train_data.drop('label', axis=1)
    train_y = train_data['label'].tolist()
    test_x = test_data.drop('label', axis=1)
    test_y = test_data['label'].tolist()
    eval_x = eval_data.drop('label', axis=1)
    eval_y = eval_data['label'].tolist()
    
    # Initialize the model.
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=args.nest, learning_rate=args.lr)
    
    # Train the model.
    model.fit(train_x, train_y)
    
    # Evaluate model on train and test data.
    train_predictions = model.predict(train_x)
    train_accuracy = accuracy_score(train_y, train_predictions)
    print(f'train accuracy: {train_accuracy:.3f}')
    test_predictions = model.predict(test_x)
    test_accuracy = accuracy_score(test_y, test_predictions)
    print(f'test accuracy: {test_accuracy:.3f}')
    eval_predictions = model.predict(eval_x)
    eval_accuracy = accuracy_score(eval_y, eval_predictions)
    print(f'eval accuracy: {eval_accuracy:.3f}')
    
    # Calculate F1 scores
    train_f1 = f1_score(train_y, train_predictions, average=None)
    test_f1 = f1_score(test_y, test_predictions, average=None)
    eval_f1 = f1_score(eval_y, eval_predictions, average=None)
    print(f'train F1 score: {train_f1[-1]:.3f}')
    print(f'test F1 score: {test_f1[-1]:.3f}')
    print(f'eval F1 score: {eval_f1[-1]:.3f}')
    
    data_frame = pd.DataFrame({
        'prediction': eval_predictions,
        'label': eval_y
    })
    
    data_frame.to_csv('adaboost_preds_no_id.csv', index=False)