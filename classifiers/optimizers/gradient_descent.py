import numpy as np


class GradientDescent:    
    def __init__(self, lr, batch_size, iters):
        '''
        lr: learning rate.
        batch_size: batch size.
        iters: iterations.
        '''
        self.lr = lr
        self.batch_size = batch_size
        self.iters = iters
    
    def optimize(self, train_data, train_labels, compute_gradient, get_parameters, update_parameters):
        '''
        train_data: n x m numpy array.
        train_labels: n x 1 numpy array.
        compute_gradient: a function that takes data and labels and returns the gradient.
        get_parameters: a function that returns parameters.
        updata_parameters: a function that takes new parameters.
        '''
        for i in range(self.iters):
            # Sample batch.
            random_sample = np.random.choice(len(train_data), self.batch_size)
            sampled_data = train_data[random_sample]
            sampled_labels = train_labels[random_sample]
            
            # Find gradient and update parameters.
            gradient = np.multiply(compute_gradient(sampled_data, sampled_labels), self.lr)
            new_params = np.subtract(get_parameters(), gradient)
            update_parameters(new_params)