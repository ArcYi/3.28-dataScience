# video link for SGD: https://www.youtube.com/watch?v=k3AiUhwHQ28
# video link for GD: https://www.youtube.com/watch?v=AeRwohPuUHQ
# reference link: https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f

import numpy as np
import matplotlib.pyplot as plt

# 瞎写的， 回头再改
def stochasticGradientDecent(X, y, m, b, step_size=0.01, epoch=100):
    n = len(y)
    for i in range(epoch):
        random_index = np.random.randint(0, n)
        X = X[random_index,:].reshape(1, X.shape[1])
        y = y[random_index].reshape(1, 1)
        for j in range(n):
            # Calculate partial derivatives
            # -2x(y - (mx + b))
            m_deriv = -2 * X[j] * (y[j] - (m * X[j] + b))
            # -2(y - (mx + b))
            b_deriv = -2 * (y[j] - (m * X[j] + b))
        m -= (m_deriv / float(n)) * step_size
        b -= (b_deriv / float(n)) * step_size
    return m, b


class LinearRegression:
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return self.w * X + self.b