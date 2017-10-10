from matplotlib import pyplot as plt
from ThucHanh_1.BaiTap_3.PrepairedData import load
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


X = load()
X = PCA(n_components=2).fit_transform(X)
y = AgglomerativeClustering(n_clusters=10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()