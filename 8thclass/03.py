import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.w1, self.b1 = np.random.randn(input_size, hidden_size) * 0.01, np.zeros((1, hidden_size))
        self.w2, self.b2 = np.random.randn(hidden_size, output_size) * 0.01, np.zeros((1, output_size))

    def forward(self, X):
        self.a1 = tanh(np.dot(X, self.w1) + self.b1)
        self.a2 = sigmoid(np.dot(self.a1, self.w2) + self.b2)
        return self.a2

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            out = self.forward(X)
            dz2, dz1 = out - y, np.dot(out - y, self.w2.T) * (self.a1 > 0)
            self.w2 -= self.lr * np.dot(self.a1.T, dz2)
            self.b2 -= self.lr * dz2.mean(axis=0)
            self.w1 -= self.lr * np.dot(X.T, dz1)
            self.b1 -= self.lr * dz1.mean(axis=0)

    def predict(self, X): return np.argmax(self.forward(X), axis=1)

df = pd.read_csv("./LAB_08_Dataset.csv").dropna()
X, y = df.iloc[:, :-1].values, OneHotEncoder(sparse_output=False).fit_transform(df.iloc[:, -1].values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)
mlp = MLP(X_train.shape[1], 10, y_train.shape[1])
mlp.train(X_train, y_train)
print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), mlp.predict(X_test)):.4f}")