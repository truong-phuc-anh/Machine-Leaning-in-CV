import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(8, 8))
nSamples = 750
randomState = random.randrange(500)
X, _ = make_blobs(n_samples=nSamples, centers=2, random_state=randomState)

# Correct number of clusters
yPredict = KMeans(n_clusters=2, random_state=randomState).fit_predict(X)


plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=yPredict)
plt.title("Bai Tap 1")

plt.show()

