import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)
k_values = range(1, 21)
accuracies = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)) for k in k_values]
plt.plot(k_values, accuracies, 'bo--')
plt.xlabel('k'), plt.ylabel('Accuracy'), plt.title('Effect of k on kNN Accuracy')
plt.axvline(1, color='r', linestyle='--', label='Overfitting (k=1)')
plt.axvline(20, color='g', linestyle='--', label='Underfitting (k=20)')
plt.legend()
plt.grid(), plt.show()
