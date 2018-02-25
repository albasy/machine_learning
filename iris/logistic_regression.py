import numpy as np


class LogisticRegression:
    def __init__(self, train_data, train_labels, num_classes):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        num_classes: number of classes.
        '''
        self.train_data = train_data
        self.train_labels = train_labels
        # Set bias.
        self.bias = 1
        
        # Store class labels in dictionary where key is a label and value is an index in self.params.
        self.classes = {}
        for i in range(len(train_labels)):
            if train_labels[i] not in self.classes:
                self.classes[train_labels[i]] = len(self.classes)
            
            if len(self.classes) == num_classes:
                break
        
        import random
        self.params = [[random.uniform(0, 1) for i in range(len(train_data[0]))] for j in range(num_classes)]
        
        # Train.
        self.train()
    
    def get_gradient(self, data, labels):
        '''
        data: n x m numpy array.
        labels: n x 1 numpy array.
        '''
        gradient = [[0 for i in range(len(data[0]))] for k in range(len(self.params))]
        for i in range(len(data)):
            for k in range(len(self.params)):
                probability = 1 / np.exp(-np.matmul(np.transpose(data[i]), self.params[k]) - self.bias)
                if k == self.classes[labels[i]]:
                    gradient[k] = (probability - 1) * data[i] + gradient[k]
                else:
                    gradient[k] = probability * data[i] + gradient[k]
        return gradient
                                    
    
    def train(self):
        '''
        Train this multi class logistic regression using stochastic gradient descent.
        '''
        for i in range(3):
            random_sample = np.random.choice(len(self.train_data), int(len(self.train_data) / 10))
            sampled_data = self.train_data[random_sample]
            sampled_labels = self.train_labels[random_sample]
            self.params = np.subtract(self.params, self.get_gradient(sampled_data, sampled_labels))


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data[:]
    y = iris.target
    
    lr = LogisticRegression(x, y, 3)
    lr.get_gradient(x[:4], y[:4])