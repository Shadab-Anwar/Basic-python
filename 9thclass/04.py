import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)
metrics = ['euclidean', 'manhattan', 'chebyshev']
accuracies = {}
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    accuracies[metric] = acc
    print(f"Accuracy using {metric} distance: {acc:.4f}")
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'red', 'green'])
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
plt.title('Impact of Distance Metrics on k-NN')
plt.ylim(0.5, 1)
plt.show()
