import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data, split, and standardize
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

# Evaluate kNN for different k values
k_values = range(1, 21)
accuracies = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)) for k in k_values]

# Print predictions for a sample test point with different k values
sample_idx = 0  # First test sample
sample_point = X_test[sample_idx].reshape(1, -1)
print("True Label:", y_test[sample_idx])

for k in [3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(f"k={k}, Predicted Class:", knn.predict(sample_point)[0])

# Plot results
plt.plot(k_values, accuracies, 'bo--')
plt.xlabel('k'), plt.ylabel('Accuracy'), plt.title('Effect of k on kNN Accuracy')
plt.grid(), plt.show()
