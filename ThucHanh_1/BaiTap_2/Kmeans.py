import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


digits = load_digits()
X = scale(digits.data)
X = PCA(n_components=2).fit_transform(X)
y = KMeans(n_clusters=10).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

