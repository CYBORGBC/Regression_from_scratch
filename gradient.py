import numpy as np
import matplotlib.pyplot as plt

# 1. Generate some synthetic data
np.random.seed(0)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)  # Random numbers between 0 and 2
true_slope = 3
true_intercept = 5
noise = np.random.randn(n_samples, 1)  # Gaussian noise

y = true_slope * X + true_intercept + noise

# 2. Add bias term (column of ones) to X
X_b = np.hstack([np.ones((n_samples, 1)), X])  # shape (100, 2)

# 3. Initialize parameters (weights) randomly
theta = np.random.randn(2, 1)  # 2 parameters: intercept and slope

# 4. Set hyperparameters
learning_rate = 0.1
n_iterations = 1000

# 5. Train using Gradient Descent
loss_history = []

for iteration in range(n_iterations):
    y_pred = X_b @ theta                # Predict
    error = y_pred - y                   # Error
    loss = (1/n_samples) * np.sum(error ** 2)  # MSE loss
    gradients = (2/n_samples) * X_b.T @ error  # Gradients
    theta = theta - learning_rate * gradients  # Update rule
    
    loss_history.append(loss)

print("Learned parameters (theta):")
print(theta)

# 6. Plot Loss Curve
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Reduction over Time')
plt.show()

# 7. Plot the data and the fitted line
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X, X_b @ theta, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit')
plt.show()
