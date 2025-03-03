import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def step_function(x):
    return 1 if x >= 0 else 0
def activation_function(x, func="sigmoid"):
    if func == "sigmoid":
        return sigmoid(x)
    elif func == "step":
        return step_function(x)
    else:
        raise ValueError("Unsupported activation function")
def train_perceptron(X_train, y_train, learning_rate, epochs, activation="sigmoid"):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    for epoch in range(epochs):
        for i in range(X_train.shape[0]):
            linear_output = np.dot(X_train[i], weights) + bias
            y_pred = activation_function(linear_output, activation)
            error = y_train[i] - y_pred
            weights += learning_rate * error * X_train[i]
            bias += learning_rate * error
    return weights, bias
def plot_decision_boundary(X, y, weights, bias, activation="sigmoid"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([activation_function(np.dot(np.array([a, b]), weights) + bias, activation) for a, b in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k')
    plt.title(f"Decision Boundary ({activation} Activation)")
    plt.show()
dataset_path = "./LAB_07_Dataset.csv"  
df = pd.read_csv(dataset_path)
X = df.iloc[:, :2].values 
y = df.iloc[:, -1].values
X_train, X_val = X[:8], X[8:]
y_train, y_val = y[:8], y[8:]
weights_step, bias_step = train_perceptron(X_train, y_train, 0.1, 10, "step")
plot_decision_boundary(X_train, y_train, weights_step, bias_step, "step")
weights_sigmoid, bias_sigmoid = train_perceptron(X_train, y_train, 0.1, 10, "sigmoid")
plot_decision_boundary(X_train, y_train, weights_sigmoid, bias_sigmoid, "sigmoid")
