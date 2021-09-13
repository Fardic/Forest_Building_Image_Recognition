import numpy as np


class metricBinary:

    def __init__(self, y_data, y_pred):
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.npv = 0
        self.specificity = 0
        self.fpr = 1
        self.fdr = 1

        self.conf_mat = [[0, 0],
                         [0, 0]]
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                if y_data[i] == 1:
                    self.conf_mat[0][0] += 1
                else:
                    self.conf_mat[0][1] += 1
            else:
                if y_data[i] == 1:
                    self.conf_mat[1][0] += 1
                else:
                    self.conf_mat[1][1] += 1

        self.performance()

    def conf_mat_print(self):
        print("           True     False")
        print("Positive", "  ", self.conf_mat[0][0], "      ", self.conf_mat[0][1])
        print("Negative", "  ", self.conf_mat[1][0], "      ", self.conf_mat[1][1],"\n")

    def performance(self):
        self.accuracy = (self.conf_mat[0][0] + self.conf_mat[1][1]) / np.sum(self.conf_mat)
        self.recall = self.conf_mat[0][0] / (self.conf_mat[0][0] + self.conf_mat[1][0])
        self.precision = self.conf_mat[0][0] / np.sum(self.conf_mat[0])
        self.npv = self.conf_mat[1][1] / np.sum(self.conf_mat[1])
        self.specificity = self.conf_mat[1][1] / (self.conf_mat[1][1] + self.conf_mat[0][1])
        self.fpr = self.conf_mat[0][1] / (self.conf_mat[1][1] + self.conf_mat[0][1])
        self.fdr = self.conf_mat[0][1] / np.sum(self.conf_mat[0])

    def perf_print(self):
        print("-----Performance Metrics-----\n")
        print("Accuracy:", self.accuracy)
        print("Recall:", self.recall)
        print("Precision:", self.precision)
        print("Negative Predictive Value:", self.npv)
        print("Specificity:", self.specificity)
        print("False Positive Rate:", self.fpr)
        print("False Discovery Rate:", self.fdr)


