import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_3.PrepairedData import load
from sklearn.preprocessing import scale

X = scale(load())
X = PCA(n_components=2).fit_transform(X)
y = DBSCAN(eps=0.6, min_samples=2).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()