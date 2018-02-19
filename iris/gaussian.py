import numpy as np


class Gaussian:
    def __init__(self, train_data, train_labels):
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
        
        # Compute parameters (label, mean, covariance, inverse of covariance, determinant of covariance) for each label.
        self.params = []
        for key in self.data_by_labels.keys():
            temp_param = [key]
            
            mean = np.sum(self.data_by_labels[key], axis=0) / len(self.data_by_labels[key])
            temp_param.append(mean)
            
            # Add bias.
            covariance = np.matmul(np.transpose(self.data_by_labels[key]), self.data_by_labels[key]) / len(self.data_by_labels[key]) + np.identity(len(self.data_by_labels[key][0])) * 0.1
            inverse_covariance = np.linalg.inv(covariance)
            det_covariance = np.linalg.det(covariance)
            temp_param.extend([covariance, inverse_covariance, det_covariance])
            
            self.params.append(temp_param)
    
    def classify(self, test_data, threshold=0):
        '''
        test_data: n x m numpy array.
        threshold: Gaussians must be more confident than this threshold.
        
        returns: n x 1 array.
        '''
        if threshold != 0:
            threshold = np.log(threshold)
        
        result = []
        for i in range(len(test_data)):
            confidences = []
            for param in self.params:
                # Compute log pdf.
                numerator = np.matmul(np.matmul(np.transpose((test_data[i] - param[1])), param[3]), (test_data[i] - param[1])) * -1 / 2
                denominator = np.log(np.sqrt(np.power((2 * np.pi), len(test_data[0])) * param[4]))
                pdf = numerator - denominator
                
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
                continue
            
            for c in range(len(confidences)):
                if confidences[c] == max_confidence:
                    result.append(self.params[c][0])
        
        return result


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    
    g = Gaussian(x, y)
    print([g.classify(x) == y])