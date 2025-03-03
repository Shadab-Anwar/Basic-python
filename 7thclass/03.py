import numpy as np
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def activation_function(x, func="sigmoid"):
    if func == "sigmoid":
        return sigmoid(x)
    elif func == "tanh":
        return tanh(x)
    elif func == "relu":
        return relu(x)
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
def predict(X, weights, bias, activation="sigmoid"):
    predictions = []
    for i in range(X.shape[0]):
        linear_output = np.dot(X[i], weights) + bias
        predictions.append(activation_function(linear_output, activation))
    return np.array(predictions)
dataset_path = "./LAB_07_Dataset.csv"
df = pd.read_csv(dataset_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_val = X[:8], X[8:]
y_train, y_val = y[:8], y[8:]
learning_rate = 0.1
epochs = 10
activation_function_type = "sigmoid"  
weights, bias = train_perceptron(X_train, y_train, learning_rate, epochs, activation_function_type)
predictions = predict(X_val, weights, bias, activation_function_type)
accuracy = np.mean(predictions.round() == y_val) * 100
print(f"Trained Weights: {weights}")
print(f"Trained Bias: {bias}")
print(f"Validation Accuracy: {accuracy:.2f}%")
