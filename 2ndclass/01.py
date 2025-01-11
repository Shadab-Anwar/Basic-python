import numpy as np

# Create a numpy array of integers from 1 to 10
array = np.arange(1, 11)

# Calculate the mean and standard deviation
mean = np.mean(array)
std_dev = np.std(array)

print(array, mean, std_dev)