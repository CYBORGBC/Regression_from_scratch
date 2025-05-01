import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate nonlinear data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 2 + 3 * X[:, 0]**2 + np.random.randn(100) * 0.2  # true relationship is quadratic

# Step 2: Create polynomial features (X, X^2)
X_poly = np.hstack([np.ones((X.shape[0], 1)), X, X**2])

# Step 3: Initialize weights
weights = np.random.randn(X_poly.shape[1])

# Hyperparameters
learning_rate = 0.05
n_epochs = 1000

# Step 4: Gradient Descent Training Loop
for epoch in range(n_epochs):
    y_pred = X_poly.dot(weights)
    error = y_pred - y
    loss = np.mean(error**2)

    gradients = 2 * X_poly.T.dot(error) / X_poly.shape[0]
    weights -= learning_rate * gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Step 5: Plot
plt.scatter(X, y, color='blue', label='Data')
x_line = np.linspace(-1, 1, 100).reshape(-1, 1)
x_line_poly = np.hstack([np.ones((x_line.shape[0], 1)), x_line, x_line**2])
y_line = x_line_poly.dot(weights)
plt.plot(x_line, y_line, color='red', label='Polynomial Fit')
plt.legend()
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
