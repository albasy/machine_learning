import numpy as np


class NaiveGaussian:
    def __init__(self, name):
        '''
        name: Name of this classifier, used when printing results etc.
        '''
        self.name = name
        self.trained = False
    
    def train(self, train_data, train_labels):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        '''
        # Sort data by labels.
        self.data_by_labels = {}
        for i in range(len(train_data)):
            if train_labels[i] not in self.data_by_labels:
                self.data_by_labels[train_labels[i]] = [train_data[i]]
            else:
                self.data_by_labels[train_labels[i]].append(train_data[i])
        
        # Compute parameters (label, mean, variance) of each feature for each label.
        self.params = []
        for key in self.data_by_labels.keys():
            mean = np.sum(self.data_by_labels[key], axis=0) / len(self.data_by_labels[key])
            variance = np.sum(np.power(self.data_by_labels[key] - mean, 2), axis=0) / len(self.data_by_labels[key])
            self.params.append([key, mean, variance])
        
        self.trained = True
    
    def classify(self, test_data, threshold=0):
        '''
        test_data: n x m numpy array.
        threshold: Gaussians must be more confident than this threshold.
        
        returns: n x 1 array.
        '''
        if not self.trained:
            return None
        
        if threshold !=  0:
            threshold = np.log(threshold)
        
        result = []
        for i in range(len(test_data)):
            confidences = []
            for param in self.params:
                # Compute log pdf.
                numerator = -np.power(test_data[i] - param[1], 2) / (2 * param[2])
                denominator = np.log(np.sqrt(2 * np.pi * param[2]))
                pdf = np.sum(numerator - denominator)
                
                if threshold == 0:
                    confidences.append(pdf)
                elif pdf >= threshold :
                    confidences.append(pdf)
                else:
                    confidences.append(0)
            
            max_confidence = max(confidences)
            # No confidence.
            if max_confidence == 0:
                result.append('unknown')
            
            for c in range(len(confidences)):
                if confidences[c] == max_confidence:
                    result.append(self.params[c][0])
                    break
        
        return result