import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


digits = load_digits()
data = digits.data
similar_data = np.corrcoef(data)

y = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)
reduced_data = PCA(n_components=2).fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.show()
