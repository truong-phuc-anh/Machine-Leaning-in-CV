import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_3.PrepairedData import load

X = load()
X = PCA(n_components=2).fit_transform(X)
y = KMeans(n_clusters=7).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


