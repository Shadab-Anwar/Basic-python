import numpy as np
import pandas as pd
def activation_function(x):
    return 1 if x >= 0 else 0
def train_perceptron(X_train, y_train, learning_rate, epochs):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    for epoch in range(epochs):
        for i in range(X_train.shape[0]):
            linear_output = np.dot(X_train[i], weights) + bias
            y_pred = activation_function(linear_output)
            error = y_train[i] - y_pred
            weights += learning_rate * error * X_train[i]
            bias += learning_rate * error
    return weights, bias
def predict(X, weights, bias):
    predictions = []
    for i in range(X.shape[0]):
        linear_output = np.dot(X[i], weights) + bias
        predictions.append(activation_function(linear_output))
    return np.array(predictions)
dataset_path = "./LAB_07_Dataset.csv" 
df = pd.read_csv(dataset_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_val = X[:8], X[8:]
y_train, y_val = y[:8], y[8:]
learning_rate = 0.1
epochs = 10
weights, bias = train_perceptron(X_train, y_train, learning_rate, epochs)
predictions = predict(X_val, weights, bias)
accuracy = np.mean(predictions == y_val) * 100
print(f"Trained Weights: {weights}")
print(f"Trained Bias: {bias}")
print(f"Validation Accuracy: {accuracy:.2f}%")
