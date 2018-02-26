from sklearn import datasets

import numpy as np

from classifiers.multi_logistic_regression import MultiLogisticRegression
from classifiers.optimizers.gradient_descent import GradientDescent

from classifiers.gaussian import Gaussian
from classifiers.naive_gaussian import NaiveGaussian


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data[:]
    y = iris.target    
    
    mlr = MultiLogisticRegression('Iris multi class logistic regression', 3, 4, np.array([1, 1, 1]))
    gradient_descent = GradientDescent(0.1, 50, 1000)
    mlr.train(x, y, gradient_descent)
    print([mlr.classify(x) == y])
    
    g = Gaussian('Iris Gaussian classifier')
    g.train(x, y)
    print([g.classify(x) == y])
    
    ng = NaiveGaussian('Iris naive Gaussian classifier')
    ng.train(x, y)
    print([ng.classify(x) == y])    