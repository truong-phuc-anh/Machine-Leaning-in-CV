import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
X = scale(digits.data)
X = PCA(n_components=2).fit_transform(X)
y = DBSCAN(eps=0.355, min_samples=10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
