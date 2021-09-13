import numpy as np
from math import log
from time import sleep
from tqdm import tqdm

class kMeans:

    clusters = []
    variance = -np.inf

    def __init__(self, k=5, iteration=100, algorithm="euclidean", random_state=0):
        self.random_state = random_state
        self.iteration = iteration
        self.algorithm = algorithm
        self.k = k
        np.random.seed(self.random_state)

    def fit(self, data):
        maximum_point = np.max(data)
        # Random Clusters are constructed
        random_clusters = []
        if self.algorithm == "euclidean":
            print("K-Means Clustering Started")
            print("----------------------------------------")
            sleep(0.1)
            for i in tqdm(range(self.iteration)):
                for j in range(self.k):
                    random_clusters.append(np.random.random(data.shape[1:]) * maximum_point)
                self.clusters = self.euclidean_distance(random_clusters, data)
                self.random_state = self.random_state + 1
                np.random.seed(self.random_state)
                random_clusters = []
        else:
            print("There is no such algorithm.")

    def euclidean_distance(self, random_clusters, data):
        # ids represent which data point close to which cluster
        ids =[]
        # k_values are the number of data points of each cluster
        k_values = np.zeros(self.k)

        for i in range(data.shape[0]):
            point_id = 0
            distance = np.inf
            for j in range(self.k):
                # Distance Algorithm
                measured_distance = abs(np.linalg.norm(random_clusters[j] - data[i]))
                if distance > measured_distance:
                    point_id = j
                    distance = measured_distance
            ids.append(point_id)
            k_values[point_id] += 1

        count = 0
        for i in k_values:
            if i == 0:
                measured_variance = -np.inf
                break
            else: count += 1
        if count == self.k:
            measured_variance = np.log(k_values)
        if self.variance >= np.sum(measured_variance):
            return self.clusters
        else:
            self.variance = np.sum(measured_variance)
            mean = []
            for i in range(self.k):
                mean.append(np.zeros(data.shape[1:]))
            for i in range(len(ids)):
                for j in range(self.k):
                    if j == ids[i]:
                        mean[j] += data[i]/k_values[j]
            return mean

    # Predicts which data belongs to which cluster
    def predict(self, data):
        prediction = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            distance = np.inf
            for j in range(self.k):
                measured_distance = abs(np.linalg.norm(self.clusters[j] - data[i]))
                if distance > measured_distance:
                    prediction[i] = j
                    distance = measured_distance
        return prediction




























