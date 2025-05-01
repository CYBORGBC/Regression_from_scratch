import numpy as np
import matplotlib.pyplot as plt

# Random seed for reproducibility
np.random.seed(42)

# Generate 50 data points
X = 2 * np.random.rand(50, 1) - 1  # Random values between [-1, 1]
y = 4 + 3 * X[:, 0] + np.random.randn(50) * 0.5  # true line plus noise

# Visualize
plt.scatter(X, y, color="blue")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Synthetic Dataset for Ridge Regression")
plt.show()
