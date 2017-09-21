'''
Duc Minh Le
PhD student
3391-3320-28
Deep Learning Entrance Exam
'''

import sys
import pickle
import math
from sklearn.decomposition import PCA

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convertRGB(image):
    return 0.299*image[:1024] + 0.587*image[1024:2048] + 0.114*image[2048:3072]

#data_1 = 'cifar-10-batches-py/data_batch_1'
K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])
data_1 = str(sys.argv[4])
dict = unpickle(data_1)


# collect samples from data
first_1k_data = dict[b'data'][:1000]
train_data = first_1k_data[N:]
test_data = first_1k_data[:N]


first_1k_label = dict[b'labels'][:1000]
train_label = first_1k_label[N:]
test_label = first_1k_label[:N]


# convert to grayscale
train_data_g = list(map(convertRGB, train_data))
test_data_g = list(map(convertRGB, test_data))


# apply PCA
pca = PCA(n_components=D, svd_solver='full')
train_data_g_pca = pca.fit_transform(train_data_g)

# implement KNN
def computeDistance(src, dst):
    distance = 0
    for index in range(D):
        distance += (src[index]-dst[index])**2
    return math.sqrt(distance)

def simple_KNN(image):
    # compute distance
    all_distances = [computeDistance(image, dst) for dst in train_data_g_pca]
    all_distances_labels = list(zip(all_distances, train_label))
    neighbors = sorted(all_distances_labels, key=lambda x: x[0])[:K]

    # voting
    vote = {}
    for index in range(K):
        vote[neighbors[index][1]] = 0
    for index in range(K):
        vote[neighbors[index][1]] += 1/neighbors[index][0]

    sortedVote = sorted(vote.items(), key=lambda x: x[1],  reverse=True)
    return(sortedVote[0])

test_data_g_pca = pca.transform(test_data_g)
n = list(map(simple_KNN, test_data_g_pca))

f = open('3391332028.txt', 'w')

for index in range(N):
    line = str(n[index][0]) + ' ' + str(test_label[index])
    f.write(line+'\n')
    print(line)


print('done')