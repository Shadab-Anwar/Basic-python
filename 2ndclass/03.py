import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10, 11)
y = x ** 2
plt.plot(x, y, label="y = x2")
plt.title("Graph of y = x2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()