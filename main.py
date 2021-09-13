# ----------------------------------------------------------------------------------------------------------------------
# Imported Libraries

import numpy as np
import cv2
import os
from time import time, sleep
from k_mean import kMeans
from log_reg import LogReg
from metrics import metricBinary
from NN import NN
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# Functions


def import_images(path):
    images = []
    colors = []
    for filename in tqdm(os.listdir(path)):
        image = canny(cv2.imread(os.path.join(path, filename)))
        image1 = cv2.imread(os.path.join(path, filename))
        sum = 0
        if image is not None:
            if len(image[:, 0]) == 150 and len(image[:, 0] == 150):
                images.append(cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA) / 255)
                sum = np.sum(image1)
                colors.append([np.sum(image1[:, :, 0]) / sum, np.sum(image1[:, :, 1]) / sum, np.sum(image1[:, :, 2]) / sum])

    return np.array(images).reshape(len(images), 100, 100), np.array(colors).reshape(len(colors), 3)


def canny(image):
    return cv2.Canny(image, 200, 200)


# ----------------------------------------------------------------------------------------------------------------------
# Read Data
# 0s represent buildings
# 1s represent forest

t1 = time()

print("Train Part of the Building Images Loading")
sleep(0.1)
train_buildings_x, train_color_build = import_images \
    ("C:/Users/fardi/Desktop/Kodlama/Python Workspace/PyCharm Workspace/StatConv/seg_train/buildings")


print("Train Part of the Forest Images Loading")
sleep(0.1)
train_forest_x, train_color_forest = import_images \
    ("C:/Users/fardi/Desktop/Kodlama/Python Workspace/PyCharm Workspace/StatConv/seg_train/forest")
train_y = np.concatenate((np.zeros(len(train_buildings_x)), np.ones(len(train_forest_x))))

print("Test Part of the Building Images Loading")
sleep(0.1)
test_buildings_x, test_color_build = import_images("C:/Users/fardi/Desktop/Ders/EEE485/Project/images/seg_test/buildings")

print("Test Part of the Forest Images Loading")
sleep(0.1)
test_forest_x, test_color_forest = import_images("C:/Users/fardi/Desktop/Ders/EEE485/Project/images/seg_test/forest")
test_y = np.concatenate((np.zeros(len(test_buildings_x)), np.ones(len(test_forest_x))))

train_x = np.concatenate((train_buildings_x, train_forest_x), axis=0)
test_x = np.concatenate((test_buildings_x, test_forest_x), axis=0)
train_x1 = np.concatenate((train_color_build, train_color_forest), axis=0)
test_x1 = np.concatenate((test_color_build, test_color_forest), axis=0)

train_x = train_x.reshape(len(train_x), 100 * 100)
test_x = test_x.reshape(len(test_x), 100 * 100)

train_x = np.append(train_x, train_x1, 1)
test_x = np.append(test_x, test_x1, 1)


# ----------------------------------------------------------------------------------------------------------------------
# K-Means Clustering Part

x = np.concatenate((train_x, test_x), axis=0)
y = np.concatenate((train_y, test_y), axis=0)

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

t2 = time()
print("\nData Import and Preprocessing Time:", (t2 - t1), "seconds.")
kmeans = kMeans(k=2, iteration=1000, random_state=10)
kmeans.fit(x)

t3 = time()
print("\nK-Means Clustering Training Time:", (t3 - t2) / 60, "minutes.")

kmeans_pred = kmeans.predict(x)
k_means_cm1 = metricBinary(y, kmeans_pred)
zeros = np.where(kmeans_pred == 0)
ones = np.where(kmeans_pred == 1)
kmeans_pred[zeros] = 1
kmeans_pred[ones] = 0
k_means_cm2 = metricBinary(y, kmeans_pred)

t4 = time()
print("\nK-Means Clustering Test Time:", (t4 - t3), "seconds.")
# ----------------------------------------------------------------------------------------------------------------------
# Logistic Regression Part

lr = LogReg(random_state=10)
lr.fit(train_x, train_y, iteration=1000, learning_rate=0.1)
t5 = time()
print("\nLogistic Regression Training Time:", (t5 - t4) / 60, "minutes.")
lr_pred = lr.pred(test_x)
lr_cm = metricBinary(test_y, lr_pred)
t6 = time()
print("\nLogistic Regression Test Time:", (t6 - t5), "minutes.")

# ----------------------------------------------------------------------------------------------------------------------
# Artificial Neural Network Part
t7 = time()
nn = NN(size=100)
nn.fit((train_x[:, :-3]), train_y, 0.001, 200)
t8 = time()
print("\nMLP Training Time:", (t8-t7) / 60, "minutes.")
nn_pred = nn.predict((test_x[:, :-3]))
t9 = time()
print("\nMLP Training Time:", (t9-t8) / 60, "minutes.")
nn_cm = metricBinary(test_y, nn_pred)

# ----------------------------------------------------------------------------------------------------------------------
# Accuracy Calculation

print("\nLogistic Regression Confusion Matrix")
lr_cm.conf_mat_print()
lr_cm.perf_print()
print("\n----------------------------------------\n ")
print("K-Means Clustering Confusion Matrix")
if k_means_cm1.accuracy > k_means_cm2.accuracy:
    k_means_cm1.conf_mat_print()
    k_means_cm1.perf_print()
else:
    k_means_cm2.conf_mat_print()
    k_means_cm2.perf_print()
print("\n----------------------------------------\n ")
print("Artificial Neural Network Confusion Matrix")
nn_cm.conf_mat_print()
nn_cm.perf_print()
print("\nTotal Time:", (time() - t1) / 60, "minutes.")
nn.plot_mse()















