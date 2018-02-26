from sklearn import datasets

from regressions.ridge_regression import RidgeRegression
from regressions.linear_regression import LinearRegression

import numpy as np


if __name__ == '__main__':
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target
    features = boston.feature_names
    
    rr = RidgeRegression('Boston ridge regression', 0.1)
    rr.train(x, y)
    classified = rr.classify(x).reshape(506, 1)
    reshaped_y = y.reshape(506, 1)
    print(np.hstack((reshaped_y, classified)))
    l1_loss = classified - reshaped_y
    print(l1_loss)
    
    lr = LinearRegression('Boston linear regression')
    lr.train(x, y)
    classified = lr.classify(x).reshape(506, 1)
    reshaped_y = y.reshape(506, 1)
    print(np.hstack((reshaped_y, classified)))
    l1_loss = classified - reshaped_y
    print(l1_loss)