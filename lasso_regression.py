import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 4 + 3 * X[:, 0] + np.random.randn(100) * 0.3

# Add bias term
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize weights
weights = np.random.randn(X_b.shape[1])

# Hyperparameters
learning_rate = 0.01
n_epochs = 1000
lambda_reg = 0.1  # L1 regularization

# Training loop
for epoch in range(n_epochs):
    y_pred = X_b.dot(weights)
    error = y_pred - y
    mse_loss = np.mean(error ** 2)
    l1_penalty = lambda_reg * np.sum(np.abs(weights[1:]))  # Exclude bias
    total_loss = mse_loss + l1_penalty

    # Gradients
    gradients = 2 * X_b.T.dot(error) / X_b.shape[0]
    gradients[1:] += lambda_reg * np.sign(weights[1:])  # L1 gradient

    weights -= learning_rate * gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Plot result
plt.scatter(X, y, color='blue')
x_line = np.linspace(-1, 1, 100)
y_line = weights[0] + weights[1] * x_line
plt.plot(x_line, y_line, color='red')
plt.title("Lasso Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
