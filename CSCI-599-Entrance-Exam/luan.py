# Entrance Exam
import sys
import numpy as np
import math
from sklearn.decomposition import PCA

K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])
path_data = sys.argv[4]
print("K = " + str(K) + "\n")
print("D = " + str(D) + "\n")
print("N = " + str(N) + "\n")
print("path_data = " + str(path_data) + "\n")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dataset = unpickle(path_data)
# print(dataset)
label = dataset[b'labels']
data = dataset[b'data']
# extract first 1000 images
data = data[0:1000]
label = label[0:1000]
# split to train and test
test = data[0:N]
train = data[N:len(data)]
test_true_label = label[0:N]
train_true_label = label[N:len(label)]

grayTrain = []
for i in range(len(train)):
    gray = []
    image = train[i]
    for j in range(1024):
        graycolor = 0.299 * image[j] + 0.587 * image[j + 1024] + 0.114 * image[j + 2 * 1024]
        gray.append(graycolor)
    grayTrain.append(gray)
grayTest = []
for i in range(len(test)):
    gray = []
    image = test[i]
    for j in range(1024):
        graycolor = 0.299 * image[j] + 0.587 * image[j + 1024] + 0.114 * image[j + 2 * 1024]
        gray.append(graycolor)
    grayTest.append(gray)

print(str(len(grayTrain)) + " * " + str(len(grayTrain[0])))

pca = PCA(n_components=D, svd_solver='full')
pca.fit(grayTrain)
print(pca.explained_variance_ratio_)
tranformed_train = pca.transform(grayTrain)
tranformed_test = pca.transform(grayTest)
print(str(len(tranformed_test)) + " * " + str(len(tranformed_test[0])))
# do K-NN for testing data
test_label = []
for i in range(N):
    distance = []
    for j in range(1000 - N):
        # compute distance to training sample j
        dist = 0
        for k in range(D):
            dist = dist + (tranformed_test[i][k] - tranformed_train[j][k]) * (
            tranformed_test[i][k] - tranformed_train[j][k])
        dist = math.sqrt(dist)
        distance.append(dist)
    # find the index of k nearest neighbor
    # print(distance)
    indice = []

    for k in range(K):
        kth_smallest_distance = sys.float_info.max
        best_index = -1
        for j in range(len(distance)):
            if (distance[j] <= kth_smallest_distance or best_index == -1) and (j not in indice):
                best_index = j
                kth_smallest_distance = distance[j]
        indice.append(best_index)
    print("Indice: ")
    print(indice)
    # find the score for each label
    score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for index in indice:
        print("distande index = " + str(distance[index]))
        score[train_true_label[index]] = score[train_true_label[index]] + 1 / (distance[index])
    best_score = score[0]
    best_index_score = 0
    for j in range(10):
        if score[j] > best_score:
            best_index_score = j
            best_score = score[j]
    test_label.append(best_index_score)

print(test_label)
output_file = open("5379845354.txt", "w")
for i in range(N):
    output_file.write(str(test_label[i]) + " " + str(test_true_label[i]) + "\n")
output_file.close()













