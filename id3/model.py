''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol

import pandas as pd

# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        pass


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        # YOUR CODE HERE
        self.most_common_label = max(set(y), key=y.count)
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        return [self.most_common_label] * len(x)



class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.tree = None
        self.default_label = None
    
    
    # Normal Cross Entropy
    def entropy(self, y):
        n = len(y)
        if n == 0:
            return 0
        counts = pd.Series(y).value_counts(normalize=True)
        return -sum(p * log2(p) for p in counts if p > 0)
    
    
    # Collision Entropy
    def collision(self, y):
        n = len(y)
        if n == 0:
            return 0
        counts = pd.Series(y).value_counts(normalize=True)
        return -log2(sum(p ** 2 for p in counts))
    
    
    # Information Gain, supports both entropy and collision entropy.
    def information_gain(self, y, left, right):
        if self.ig_criterion == 'entropy':
            return self.entropy(y) - (len(left) / len(y)) * self.entropy(left) - (len(right) / len(y)) * self.entropy(right)
        elif self.ig_criterion == 'collision':
            return self.collision(y) - (len(left) / len(y)) * self.collision(left) - (len(right) / len(y)) * self.collision(right)
        
    
    # Pick the best attribute for the branching.
    def choose_attribute(self, x, y):
        best_ig = 0
        best_feature = None
        best_threshold = None
        
        for feature in x.columns:
            for threshold in x[feature].unique():
                # Separates the values into two groups to fit into information gain.
                left = y[x[feature] == threshold]
                right = y[x[feature] != threshold]
                ig = self.information_gain(y, left, right)
                
                # Replace the best information gain if the current one is better.
                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    
    # Recursive function to build the decision tree.
    def id3(self, x, y, depth=0):
        # If there is only a root node or other niche conditions, return the most common label.
        if len(set(y)) == 1 or (self.depth_limit is not None and depth >= self.depth_limit):
            return {'label': y.mode()[0]}
        
        best_feature, best_threshold = self.choose_attribute(x, y)
        # If all labels are the same, return the most common label.
        if best_feature is None:
            return {'label': y.mode()[0]}
        
        dictionary = {
            'feature': best_feature,
            'threshold': best_threshold
        }
        # Adapts to n number of features available.
        for value in x[best_feature].unique():
            mask = x[best_feature] == value
            dictionary[value] = self.id3(x[mask], y[mask], depth + 1)
            
        return dictionary


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''

        # YOUR CODE HERE
        self.tree = self.id3(x, pd.Series(y))
        self.default_label = pd.Series(y).mode()[0]
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        predictions = []
        for _, row in x.iterrows():
            tree = self.tree
            while 'label' not in tree:
                feature = tree['feature']
                # value = row[feature]
                value = row.get(feature, None)
                if value in tree:
                    tree = tree[value]
                else:
                    # If the value is not in the tree, return the most common label.
                    tree = {'label': self.default_label}
                    
            predictions.append(tree['label'])
            
        return predictions