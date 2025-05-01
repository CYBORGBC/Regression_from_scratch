import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 2 + 3 * X[:, 0]**2 + np.random.randn(100) * 0.2

# Split data: 80% train, 20% test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Polynomial features (X, X^2)
def poly_features(X):
    return np.hstack([np.ones((X.shape[0], 1)), X, X**2])

X_train_poly = poly_features(X_train)
X_test_poly = poly_features(X_test)

# Train model using training data
weights = np.random.randn(X_train_poly.shape[1])
learning_rate = 0.05
n_epochs = 1000

for epoch in range(n_epochs):
    y_pred_train = X_train_poly.dot(weights)
    error = y_pred_train - y_train
    loss = np.mean(error**2)

    gradients = 2 * X_train_poly.T.dot(error) / X_train_poly.shape[0]
    weights -= learning_rate * gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Loss: {loss:.4f}")

# Evaluate on test set
y_pred_test = X_test_poly.dot(weights)
test_loss = np.mean((y_pred_test - y_test) ** 2)
print(f"Test Loss: {test_loss:.4f}")

# Plot
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
x_line = np.linspace(-1, 1, 100).reshape(-1, 1)
x_line_poly = poly_features(x_line)
y_line = x_line_poly.dot(weights)
plt.plot(x_line, y_line, color='red', label='Model Fit')
plt.legend()
plt.title("Train/Test Split in Polynomial Regression")
plt.show()
