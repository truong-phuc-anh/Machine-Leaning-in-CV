from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
from sklearn.decomposition import PCA


digits = datasets.load_digits()
X = digits.data
X = PCA(n_components=2).fit_transform(X)
y = AgglomerativeClustering(n_clusters=10).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()