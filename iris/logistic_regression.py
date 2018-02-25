import numpy as np


class LogisticRegression:
    def __init__(self, train_data, train_labels, num_classes, iterations):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        num_classes: number of classes.
        '''
        self.train_data = train_data
        self.train_labels = train_labels
        # Set bias.
        self.bias = 1
        
        # Store class labels and corresponding index in self.params.
        self.label_to_index = {}
        self.index_to_label = {}
        for i in range(len(train_labels)):
            if train_labels[i] not in self.label_to_index:
                self.label_to_index[train_labels[i]] = len(self.label_to_index)
                self.index_to_label[len(self.label_to_index) - 1] = train_labels[i]
            
            if len(self.label_to_index) == num_classes:
                break
        
        import random
        self.params = [[random.uniform(0, 1) for i in range(len(train_data[0]))] for j in range(num_classes)]
        
        # Train.
        self.train(iterations)
    
    def get_gradient(self, data, labels):
        '''
        data: n x m numpy array.
        labels: n x 1 numpy array.
        '''
        gradient = [[0 for i in range(len(data[0]))] for k in range(len(self.params))]
        for i in range(len(data)):
            for k in range(len(self.params)):
                probability = 1 / (1 + np.exp(-np.matmul(np.transpose(data[i]), self.params[k]) - self.bias))
                if k == self.label_to_index[labels[i]]:
                    gradient[k] = (probability - 1) * data[i] + gradient[k]
                else:
                    gradient[k] = probability * data[i] + gradient[k]
        return gradient
                                    
    
    def train(self, iterations):
        '''
        Train this multi class logistic regression using stochastic gradient descent.
        '''
        for i in range(iterations):
            random_sample = np.random.choice(len(self.train_data), int(len(self.train_data) / 10))
            sampled_data = self.train_data[random_sample]
            sampled_labels = self.train_labels[random_sample]
            self.params = np.subtract(self.params, np.multiply(self.get_gradient(sampled_data, sampled_labels), 0.1))
    
    def classify(self, data):
        '''
        data: n x m numpy array.
        
        returns: n x 1 array.
        '''
        result = []
        for i in range(len(data)):
            probabilities = np.exp(np.matmul(self.params, data[i]) + self.bias)
            max_probability = max(probabilities)
            for p in range(len(probabilities)):
                if probabilities[p] == max_probability:
                    result.append(self.index_to_label[p])
                    break
        return result


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data[:]
    y = iris.target
    
    lr = LogisticRegression(x, y, 3, 1000)
    # Print loss.
    print([lr.classify(x) == y])