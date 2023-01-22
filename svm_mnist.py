from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
import time

START_TIME = time.time()
print("Fetching data...")
dataset = fetch_mldata('MNIST original', data_home='MNIST_DATA')
print("DONE! fetching data")

features = dataset.data 
labels = dataset.target
print('Duration of fetching data: ' + str(round((time.time() - START_TIME), 2))+'s')

TRAIN_PERCENT = 0.1
num_data_samples = len(features)

num_train = int(TRAIN_PERCENT * num_data_samples)

train_indices = np.random.choice(np.arange(num_train), num_train).tolist()
test_indices = np.array([ele for ele in range(num_data_samples) if ele not in train_indices])


pp = preprocessing.StandardScaler().fit(features)
features = pp.transform(features)

print("Reducing dimension...")
pca = PCA(n_components=50)
features = pca.fit_transform(features)


train_features = features[train_indices]
train_labels = labels[train_indices]
test_features = features[test_indices]
test_labels = labels[test_indices]


print("Training SVM...")
clf = SVC(C=5,gamma=.05)
clf.fit(train_features, train_labels)
print("DONE! Training SVM!")


train_score = clf.score(train_features, train_labels)
test_score = clf.score(test_features, test_labels)

print("Train acc: {:.4f}, test acc: {:.4f}".format(train_score, test_score))
 

print('Total duration: ' + str(round((time.time() - START_TIME), 2))+'s')