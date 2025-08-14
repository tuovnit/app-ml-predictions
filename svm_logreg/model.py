''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol

import numpy as np

from utils import clip, shuffle_data



# set the numpy random seed so our randomness is reproducible
np.random.seed(1)

MODEL_OPTIONS = ['majority_baseline', 'svm', 'logistic_regression']


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the classes further down.
class Model(Protocol):
    def __init__(**hyperparam_kwargs):
        ...

    def get_hyperparams(self) -> dict:
        ...

    def loss(self, ) -> float:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE
        self.most_common_label = None


    def get_hyperparams(self) -> dict:
        return {}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        return None
    

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''
    
        # YOUR CODE HERE
        labels, counts = np.unique(y, return_counts=True)
        self.most_common_label = labels[np.argmax(counts)]


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        return [self.most_common_label] * len(x)
    


class SupportVectorMachine(Model):
    def __init__(self, num_features: int, lr0: float, C: float):
        '''
        Initialize a new SupportVectorMachine model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            C (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.C = C

        # YOUR CODE HERE
        self.num_features = num_features
        self.w = np.zeros(num_features)  # Initialize weights to zero
        self.t = 0  # Initialize the time step for learning rate decay
        self.losses = []  # List to store losses for each epoch, for plotting.

    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'C': self.C}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Don't forget to convert the {0, 1} label to {-1, 1}.
        '''

        # YOUR CODE HERE
        y_i = np.where(y_i == 0, -1, 1) # Convert label to {-1, 1}.
        margin = 1 - y_i * np.dot(self.w, x_i)
        hinge = np.maximum(0, margin)
        # regularization = 0.5 * np.dot(self.w, self.w)
        # loss = np.mean(hinge) * self.C
        return hinge
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        y = np.where(y == 0, -1, 1)
        
        for epoch in range(epochs):
            x, y = shuffle_data(x, y)
            losses = [] # For plotting.
            
            for i in range(len(x)):
                margin = y[i] * np.dot(self.w, x[i])
                gamma_t = self.lr0 / (1 + self.t)   # Learning rate decay.
                
                if margin <= 1:
                    self.w = (1 - gamma_t) * self.w + gamma_t * self.C * y[i] * x[i]
                    loss = 1 - margin   # For plotting.
                else:   
                    self.w = (1 - gamma_t) * self.w
                    loss = 0    # For plotting.
                    
                losses.append(loss) # For plotting.
                
            self.t += 1
            self.losses.append(np.mean(losses)) # For plotting.


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        predictions = np.dot(x, self.w)
        return [1 if pred > 0 else 0 for pred in predictions]


class LogisticRegression(Model):
    def __init__(self, num_features: int, lr0: float, sigma2: float):
        '''
        Initialize a new LogisticRegression model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            sigma2 (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.sigma2 = sigma2

        # YOUR CODE HERE
        self.w = np.zeros(num_features)  # Initialize weights to zero.
        self.t = 0  # Initialize the time step.
        self.losses = []  # List to store losses for each epoch, for plotting.

    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'sigma2': self.sigma2}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
                If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
        '''

        # YOUR CODE HERE
        y_i = np.where(y_i == 0, -1, 1) # Convert label to {-1, 1}.
        z = np.dot(x_i, self.w)
        z = clip(z, 100)
        logistic = np.log(1 + np.exp(-y_i * z))
        return logistic
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        y = np.where(y == 0, -1, 1)
        for epoch in range(epochs):
            x, y = shuffle_data(x, y)
            losses = []
            
            for i in range(len(x)):
                z = np.dot(-y[i], np.dot(x[i], self.w))
                sig = sigmoid(z)
                loss = np.dot(-y[i], np.dot(x[i], sig))
                grad = loss + (2 / self.sigma2) * self.w
            
                gamma_t = self.lr0 / (1 + self.t)   # Learning rate decay.
                self.w = self.w - gamma_t * grad
                losses.append(loss) # For plotting.
                                
            self.t += 1
            self.losses.append(np.mean(losses)) # For plotting.
            

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        predictions = np.dot(x, self.w)
        probabilities = sigmoid(predictions)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]


def sigmoid(z: float) -> float:
    '''
    The sigmoid function.

    Args:
        z (float): the argument to the sigmoid function.

    Returns:
        float: the sigmoid applied to z.

    Hints:
        - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
            If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
          
    '''
    
    # YOUR CODE HERE
    return 1 / (1 + np.exp(-clip(z, 100)))