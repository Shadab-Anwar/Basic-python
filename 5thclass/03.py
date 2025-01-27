from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Cross-Validation Scores: {results}")
print(f"Mean Accuracy: {(results.mean()):.2f}")