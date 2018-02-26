from .functions.softmax import Softmax

import numpy as np


class MultiLogisticRegression:
    def __init__(self, name, num_classes, num_features, biases):
        '''
        name: Name of this classifier, used when printing results etc.
        num_classes: total number of distinct classes.
        num_features: number of features.
        biases: k x 1 numpy array.
        '''
        self.name = name
        self.func = Softmax(num_classes, num_features, biases)
        self.trained = False
    
    def train(self, train_data, train_labels, optimizer):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        optimizer: optimizer for training.
        
        Train this multi class logistic regression with the given optimizer and train data.
        '''        
        optimizer.optimize(train_data, train_labels, self.func.compute_gradient, self.func.get_parameters, self.func.update_parameters)
        self.trained = True
    
    def classify(self, data):
        '''
        data: n x m numpy array.
        
        returns: n x 1 array.
        '''
        if not self.trained:
            return None
        
        probabilities = self.func.compute(data)
        result = []
        for i in range(len(probabilities)):
            max_probability = max(probabilities[i])
            for p in range(len(probabilities[i])):
                if probabilities[i][p] == max_probability:
                    result.append(p)
        
        return result