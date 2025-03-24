import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Load and preprocess data
iris = datasets.load_iris()
X = StandardScaler().fit_transform(iris.data)
y_true = iris.target

# Clustering
y_dbscan = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
y_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

# Evaluation
print(f'ARI - DBSCAN: {adjusted_rand_score(y_true, y_dbscan):.3f}')
print(f'ARI - K-Means: {adjusted_rand_score(y_true, y_kmeans):.3f}')

# Visualization
X_pca = PCA(n_components=2).fit_transform(X)
titles = ['Actual', 'DBSCAN', 'K-Means']
labels = [y_true, y_dbscan, y_kmeans]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, title, label in zip(axes, titles, labels):
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis', edgecolor='k')
    ax.set_title(title)
plt.show()
