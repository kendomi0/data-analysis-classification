import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

datasets_dict = {
    "noisy_circles": datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8),
    "noisy_moons": datasets.make_moons(n_samples=1500, noise=0.05, shuffle=True, random_state=42),
    "blobs": datasets.make_blobs(n_samples=1200, n_features=2,centers=3, cluster_std=1.2, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=False),
    "anisotropic": (np.dot(datasets.make_blobs(n_samples=1500, centers=2, random_state=42, shuffle=True, center_box=(-10.0, 10.0))[0], [[0.6, -0.6], [-0.4, 0.8]]), datasets.make_blobs(n_samples=1500, centers=2, random_state=42, shuffle=True, center_box=(-10.0, 10.0))[1]),
    "varied": datasets.make_classification(n_samples=1500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
}

base_x, base_y = datasets.make_blobs(n_samples=1500, centers=2, random_state=42, shuffle=True, center_box=(-10.0, 10.0))

# Adjusting anisotropic data to ensure the transformation is correctly applied
new_x = np.dot(base_x, [[0.6, -0.6], [-0.4, 0.8]])
datasets_dict["anisotropic"] = (new_x, base_y)

# Change this depending on which dataset you want to use
current_dataset = input("Type in which dataset you'd like to use: ")

X, y = datasets_dict[current_dataset]


# NORMALIZATION
# Min-max normalization
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
# Z-score normalization
zscore = StandardScaler()
X_zscore = zscore.fit_transform(X)


# CLASSIFIERS


# ### Naive bayes classifier
print("### Naive bayes classifier")

# LOO unnormalized (NB)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy unnormalized (NB): {round(np.mean(scores), 3)}")
# LOO minmax (NB)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_minmax):
    X_train, X_test = X_minmax[train_index], X_minmax[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy minmax (NB): {round(np.mean(scores), 3)}")
# LOO zscore (NB)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_zscore):
    X_train, X_test = X_zscore[train_index], X_zscore[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy zscore (NB): {round(np.mean(scores), 3)}")


# ### Decision tree
print("### Decision tree")
# LOO unnormalized (DT)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy unnormalized (DT): {round(np.mean(scores), 3)}")

# LOO minmax (DT)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_minmax):
    X_train, X_test = X_minmax[train_index], X_minmax[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy minmax (DT): {round(np.mean(scores), 3)}")

# LOO zscore (DT)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_zscore):
    X_train, X_test = X_zscore[train_index], X_zscore[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy zscore (DT): {round(np.mean(scores), 3)}")


# ### Support vector machines
print("### Support vector machines")
# LOO unnormalized (SVM)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy unnormalized (SVM): {round(np.mean(scores), 3)}")
# LOO minmax (SVM)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_minmax):
    X_train, X_test = X_minmax[train_index], X_minmax[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy minmax (SVM): {round(np.mean(scores), 3)}")
# LOO zscore (SVM)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_zscore):
    X_train, X_test = X_zscore[train_index], X_zscore[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy zscore (SVM): {round(np.mean(scores), 3)}")


# ### K-nearest-neighbor
print("### K-nearest-neighbor")
# LOO unnormalized (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    leave_one = LeaveOneOut()
    scores = []
    for train_index, test_index in leave_one.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
best_k = max(k_scores, key=k_scores.get)
print(f"({current_dataset}) The best k value for KNN with LOO unnormalized is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# LOO minmax (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    leave_one = LeaveOneOut()
    scores = []
    for train_index, test_index in leave_one.split(X_minmax):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
best_k = max(k_scores, key=k_scores.get)
print(f"({current_dataset}) The best k value for KNN with LOO minmax is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# LOO zscore (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    leave_one = LeaveOneOut()
    scores = []
    for train_index, test_index in leave_one.split(X_zscore):
        X_train, X_test = X_zscore[train_index], X_zscore[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
best_k = max(k_scores, key=k_scores.get)
print(f"({current_dataset}) The best k value for KNN with LOO zscore is {best_k} with accuracy {round(k_scores[best_k], 3)}")



# ### Artificial neural networks
# Use a sample because LOO with ANN takes very long to run
sample_size = 200
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[indices]
y_sample = y[indices]

minmax_sample = MinMaxScaler()
X_sample_minmax = minmax_sample.fit_transform(X_sample)

zscore_sample = StandardScaler()
X_sample_zscore = zscore_sample.fit_transform(X_sample)

print("### Artificial neural networks")
# LOO unnormalized (ANN)

leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_sample):
    X_train, X_test = X_sample[train_index], X_sample[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(20, 20, 10), activation='relu', max_iter=1500, random_state=42, early_stopping=True, validation_fraction=0.1,  alpha=0.01,)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy unnormalized (ANN): {round(np.mean(scores), 3)}")
# LOO minmax (ANN)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_sample_minmax):
    X_train, X_test = X_sample_minmax[train_index], X_sample_minmax[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(20, 20, 10), activation='relu', max_iter=1500, random_state=42, early_stopping=True, validation_fraction=0.1,  alpha=0.01,)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy minmax (ANN): {round(np.mean(scores), 3)}")
# LOO zscore (ANN)
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_sample_zscore):
    X_train, X_test = X_sample_zscore[train_index], X_sample_zscore[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(20, 20, 10), activation='relu', max_iter=1500, random_state=42, early_stopping=True, validation_fraction=0.1, alpha=0.01,)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Leave-one-out accuracy zscore (ANN): {round(np.mean(scores), 3)}")
