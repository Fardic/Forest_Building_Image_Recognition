import numpy as np
from time import sleep
from tqdm import tqdm

class LogReg:

    def __init__(self, random_state=0):
        self.b = np.random.rand()
        self.w = np.random.rand()
        self.random_state = random_state
        np.random.seed(self.random_state)
        old_settings = np.seterr(all='ignore')

    def fit(self, x, y, iteration=10, learning_rate=0.01):
        self.w = np.random.random_sample(len(x[0]))
        print("Logistic Regression is Training the Data:")
        print("----------------------------------------")
        sleep(0.1)
        for i in tqdm(range(iteration)):
            y_pred = self.pred(x)
            grad = np.sum(np.subtract(y, y_pred))
            self.b += learning_rate * grad
            grad_w = np.dot(np.subtract(y, y_pred).T, x)
            self.w += learning_rate * grad_w

    def pred(self, x):
        pred = []
        for i in x:
            if self.sigmoid_func(np.dot(self.w.T, i) + self.b) > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))


