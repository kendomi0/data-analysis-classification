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

base_x, base_y = datasets.make_blobs(n_samples=1500, centers=2, random_state=42,shuffle=True, center_box=(-10.0, 10.0))

# Adjusting anisotropic data to ensure the transformation is correctly applied
new_x = np.dot(base_x, [[0.6, -0.6], [-0.4, 0.8]])
datasets_dict["anisotropic"] = (new_x, base_y)

# Change this depending on which dataset you want to use
current_dataset = "noisy_circles"

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
# Random subsampling unnormalized (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (NB): {round(np.mean(scores), 3)}")
# Random subsampling minmax (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (NB): {round(np.mean(scores), 3)}")
# Random subsampling zscore (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (NB): {round(np.mean(scores), 3)}")


# ### Decision tree
print("### Decision tree")
# Random subsampling unnormalized (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (DT): {round(np.mean(scores), 3)}")
# Random subsampling minmax (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (DT): {round(np.mean(scores), 3)}")
# Random subsampling zscore (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (DT): {round(np.mean(scores), 3)}")


# ### Support vector machines
print("### Support vector machines")
# Random subsampling unnormalized (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (SVM): {round(np.mean(scores), 3)}")
# Random subsampling minmax (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (SVM): {round(np.mean(scores), 3)}")
# Random subsampling zscore (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (SVM): {round(np.mean(scores), 3)}")


# ### K-nearest-neighbor
print("### K-nearest-neighbor")
# Random subsampling unnormalized (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (KNN with k={k}): {round(k_scores[k], 3)}")

# Random subsampling minmax (KNN)
k_vals = [3, 5, 7]
k_scores = {}

for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling minmax (KNN with k={k}): {round(k_scores[k], 3)}")
# Random subsampling zscore (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling zscore (KNN with k={k}): {round(k_scores[k], 3)}")


# ### Artificial neural networks
print("### Artificial neural networks")
# Random subsampling unnormalized (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (ANN): {round(np.mean(scores), 3)}")
# Random subsampling minmax (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (ANN): {round(np.mean(scores), 3)}")
# Random subsampling zscore (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (ANN): {round(np.mean(scores), 3)}")

