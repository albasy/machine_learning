import numpy as np


class Softmax():
    def __init__(self, num_classes, num_features, biases):
        '''
        num_classes: total number of disctinct classes.
        num_features: number of features.
        biases: k x 1 numpy array.
        '''
        # Initialize parameters with random values.
        import random
        self.params = [[random.uniform(0, 1) for i in range(num_features)] for j in range(num_classes)]
        
        self.biases = biases
    
    def get_parameters(self):
        '''
        returns: k x m numpy array.
        '''
        return self.params
    
    def update_parameters(self, new_params):
        '''
        new_params: k x m numpy array.
        '''
        self.params = new_params
    
    def compute(self, data):
        '''
        data: n x m numpy array.
        
        returns: n x k array.
        '''
        unormalized_probabilities = np.exp(np.matmul(data, np.transpose(self.params)) + self.biases)
        return unormalized_probabilities / np.reshape(np.sum(unormalized_probabilities, axis=1), (len(unormalized_probabilities), 1))
    
    def compute_gradient(self, data, labels):
        """
        data: n X m numpy array.
        labels: n X 1 numpy array.
        
        Returns k x d array.
        """
        # Compute probabilities.
        probabilities = self.compute(data)
        
        # One hot encoding for class labels.
        one_hot_labels = []
        for i in range(len(labels)):
            label_vector = [0 for j in range(len(self.params))]
            label_vector[labels[i]] = 1
            one_hot_labels.append(label_vector)
        
        # Compute loss.
        loss = probabilities - one_hot_labels
        
        # Compute loss x input.
        return np.matmul(np.transpose(loss), data)
