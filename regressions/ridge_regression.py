import numpy as np


class RidgeRegression:
    def __init__(self, name, penalty):
        '''
        name: Name of this classifier, used when printing results etc.
        penalty: complexity penalty.
        '''
        self.name = name
        self.penalty = penalty
        self.trained = False
    
    def train(self, train_data, train_labels):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        '''
        # Add bias.
        self.train_data = np.hstack((train_data, [[1] for i in range(len(train_data))]))
        self.train_labels = train_labels
        
        XTX = np.linalg.inv(np.matmul(np.transpose(self.train_data), self.train_data) + np.identity(len(self.train_data[0])) * self.penalty)
        XTY = np.matmul(np.transpose(self.train_data), train_labels)
        # (lambdaI + X^TX)^-1X^TY.
        self.params = np.matmul(XTX, XTY)
        
        self.trained = True
    
    def classify(self, test_data):
        '''
        test_data: n x m numpy array.
        
        returns: n x 1 array.
        '''
        if not self.trained:
            return None
        
        return np.matmul(np.hstack((test_data, [[1] for i in range(len(test_data))])), self.params)