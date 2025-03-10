import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation=tanh, lr=0.01):
        self.lr, self.activation = lr, activation
        self.w1, self.b1, self.w2, self.b2 = [np.random.randn(*s) * 0.01 for s in [(input_size, hidden_size), (1, hidden_size), (hidden_size, output_size), (1, output_size)]]
    
    def forward(self, X):
        self.a1 = self.activation(np.dot(X, self.w1) + self.b1)
        return sigmoid(np.dot(self.a1, self.w2) + self.b2)
    
    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            out, dz2 = self.forward(X), self.forward(X) - y
            dz1 = np.dot(dz2, self.w2.T) * (self.a1 > 0)
            for w, b, dz in [(self.w2, self.b2, dz2), (self.w1, self.b1, dz1)]:
                w -= self.lr * np.dot((self.a1 if w is self.w2 else X).T, dz)
                b -= self.lr * dz.mean(axis=0)
    
    def predict(self, X): return np.argmax(self.forward(X), axis=1)

def plot_decision_boundary(X, y, model):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100), np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    plt.contourf(xx, yy, model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='k', cmap='coolwarm')
    plt.show()

df = pd.read_csv("./LAB_08_Dataset.csv").dropna()
X, y = df.iloc[:, :2].values, OneHotEncoder(sparse_output=False).fit_transform(df.iloc[:, -1].values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

for activation in [tanh, relu]:
    mlp = MLP(X_train.shape[1], 10, y_train.shape[1], activation)
    mlp.train(X_train, y_train)
    print(f"Accuracy ({activation.__name__}): {accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test)):.4f}")
    plot_decision_boundary(X_train, y_train, mlp)
