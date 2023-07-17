import numpy as np

class AdalineGD:
    def __init__(self, learning_rate=0.01, iteration=50, random_state=1):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []
        for _ in range(self.iteration):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta*2.00*X.T.dot(errors)/X.shape[0]
            self.b_ += self.eta*2.00*errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        # Insert computation here if activation function is given, if not just return X
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0, 1, 0)