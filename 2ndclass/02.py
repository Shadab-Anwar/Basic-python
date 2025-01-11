import numpy as np
matrix = np.array([[1, 2],
                   [4, 5],])
inverse=np.linalg.inv(matrix)
print("Original matrix: ")
print(matrix)
print("Inverse matrix: ")
print(inverse)