from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(model, X, y, cv=skfold)
print(f"Stratified Cross-Validation Scores: {results}")
print(f"Mean Accuracy: {results.mean()}")