import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ThucHanh_1.BaiTap_3.PrepairedData import load

data = load()
similar_data = cosine_similarity(data)

y = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)
reduced_data = PCA(n_components=2).fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.show()