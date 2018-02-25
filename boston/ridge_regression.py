import numpy as np


class RidgeRegression:
    def __init__(self, train_data, train_labels, penalty=0.1):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        '''
        # Add bias.
        self.train_data = np.hstack((train_data, [[1] for i in range(len(train_data))]))
        self.train_labels = train_labels
        
        XTX = np.linalg.inv(np.matmul(np.transpose(self.train_data), self.train_data) + np.identity(len(self.train_data[0])) * penalty)
        XTY = np.matmul(np.transpose(self.train_data), train_labels)
        # (lambdaI + X^TX)^-1X^TY.
        self.params = np.matmul(XTX, XTY)
    
    def classify(self, test_data):
        '''
        test_data: n x m numpy array.
        
        returns: n x 1 array.
        '''
        return np.matmul(np.hstack((test_data, [[1] for i in range(len(test_data))])), self.params)


if __name__ == '__main__':
    from sklearn import datasets
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target
    features = boston.feature_names
    
    lr = RidgeRegression(x, y)
    classified = lr.classify(x).reshape(506, 1)
    reshaped_y = y.reshape(506, 1)
    print(np.hstack((reshaped_y, classified)))
    
    l1_loss = classified - reshaped_y
    print(l1_loss)