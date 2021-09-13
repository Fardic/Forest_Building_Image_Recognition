import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt


# 1 Deep Layer Neural Network Class
class NN:

    def __init__(self, size):
        # Number of Perceptron in the Hidden Layer
        self.size = size

        # Initialization of Weights (Bias Terms Are Embedded)
        self.w1 = np.random.random(2 * self.size).reshape(2, self.size)
        self.w2 = np.random.random(self.size + 1).reshape(self.size + 1, 1)
        self.w1_grad = np.random.random(2 * self.size).reshape(2, self.size)
        self.w2_grad = np.random.random(self.size + 1)

        # Some MLP calculation variables that is used in multiple functions
        self.v = np.random.random(self.size).reshape(self.size, 1)
        self.o = np.random.random(self.size).reshape(self.size, 1)
        self.v2 = np.random.random(1)
        self.pred = np.random.random(1)

        # Mse array that holds mse of the model for every iteration
        self.mse = []
        # Number of Iterations in Training
        self.iteration = 0

    # Function trains the model according to training dataset
    # x_train: All samples of training dataset
    # y_train: All labels of training dataset
    # lr: Learning Rate
    # iterations: Number of Iteration for Model to optimize weights
    def fit(self, x_train, y_train, lr, iterations):
        # Cost initialization
        error = np.inf
        self.iteration = iterations

        # Bias perceptron is added to the Input Layer
        x = np.hstack((x_train, np.ones((len(x_train), 1))))
        # Re-Initialization of weights of first layer (Since number of features in dataset will be known in fit function)
        self.w1 = np.random.random((len(x_train[0]) + 1) * self.size).reshape(len(x_train[0]) + 1, self.size)
        self.w1_grad = np.random.random((len(x_train[0]) + 1) * self.size).reshape(len(x_train[0]) + 1, self.size)
        # For every iteration, Forward and Backward Propagation and MSE calculation
        print("ANN Training Started")
        print("----------------------------------------")
        sleep(0.1)
        for i in tqdm(range(iterations)):
            self.forward_prop(x)
            error = np.subtract(y_train.reshape(len(y_train), 1), self.pred)
            self.mse.append(np.dot(error.T, error)[0] / len(y_train))
            self.backward_prop(x, y_train, lr)

    # Function predicts labels of given samples according to present weights
    def forward_prop(self, x):
        # Input Layer -> Hidden Layer
        self.v = np.dot(x, self.w1)
        self.o = self.sigmoid(self.v)
        # Bias perceptron is added to the Hidden Layer
        self.o = np.hstack((self.o, np.ones((len(self.o), 1))))
        # Hidden Layer -> Output Layer
        self.v2 = np.dot(self.o, self.w2)
        self.pred = self.sigmoid(self.v2)
        self.pred = self.pred.reshape(len(self.pred), 1)

    # Function optimizes weights with Full Batch Gradient Descent
    def backward_prop(self, x_sample, y_sample, lr):
        # Gradient Calculation of weights that connect Hidden and Output Layer
        # Gradient = (Output of Hidden Layer) . (-2 * (y - prediction) * σ(prediction) (1 - σ(prediction)))
        self.w2_grad = np.dot(self.o.T, (-2 * (y_sample.reshape(len(y_sample), 1) - self.pred) *
                                         np.multiply(self.sigmoid(self.pred), 1 - self.sigmoid(self.pred))))

        # Gradient Calculation of weights that connect Input and Hidden Layer
        # Gradient = (Input) . ((-2 * (y - prediction) * σ(prediction) (1 - σ(prediction))) * (σ(Output of Hidden Layer) (1 - σ(Output of Hidden Layer))))
        self.w1_grad = np.dot(x_sample.T,  (np.dot(-2*(y_sample.reshape(len(y_sample), 1) - self.pred) *
                                                   np.multiply(self.sigmoid(self.pred), 1 - self.sigmoid(self.pred)),
                                                   self.w2[:-1].reshape(len(self.w2[:-1]), 1).T) *
                                            np.multiply(self.sigmoid(self.o[:, :-1]), 1 - self.sigmoid(self.o[:, :-1]))))

        # Calculated Gradients are subtracted from current weights
        self.w2 -= np.dot(lr, self.w2_grad)
        self.w1 -= np.dot(lr, self.w1_grad)

    # Function returns predictions according to given samples
    def predict(self, x_test):
        predictions = []
        x = np.hstack((x_test, np.ones((len(x_test), 1))))
        self.forward_prop(x)
        for i in range(len(self.pred)):
            if self.pred[i] < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions

    # Function plots the MSE values due to every iteration
    def plot_mse(self):
        plt.plot(list(range(1, self.iteration + 1)), self.mse)
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.title("MSE vs Iteration Plot for ANN")
        plt.show()

    # Function returns sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


