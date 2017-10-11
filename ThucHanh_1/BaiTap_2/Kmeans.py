# Apply Kmeans on hand digits dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 30/09/2017


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data & reshape feature space by PCA
digits = load_digits()
X = scale(digits.data)
X = PCA(n_components=2).fit_transform(X)

# Apply Kmeans
y = KMeans(n_clusters=10).fit_predict(X)

# Visualize result
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 2 - Kmeans')
plt.show()

