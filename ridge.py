import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
X = 2 * np.random.rand(50, 1) - 1  # Random values between [-1, 1]
y = 4 + 3 * X[:, 0] + np.random.randn(50) * 0.5  # true line plus noise


# Add bias term (X0 = 1)
X_b = np.hstack([np.ones((X.shape[0], 1)), X])
print(X_b[:5])
# Initialize weights
n_features = X_b.shape[1]
weights = np.random.randn(n_features)

# Hyperparameters
learning_rate = 0.01
n_epochs = 1000
lambda_reg = 0.1  # Regularization strength

# Training loop
for epoch in range(n_epochs):
    y_pred = X_b.dot(weights)
    error = y_pred - y
    mse_loss = (error ** 2).mean()
    l2_penalty = lambda_reg * np.sum(weights[1:] ** 2)  # exclude bias term from regularization
    total_loss = mse_loss + l2_penalty
    
    gradients = 2 * X_b.T.dot(error) / X.shape[0]
    gradients[1:] += 2 * lambda_reg * weights[1:]  # add L2 term except for bias
    
    weights -= learning_rate * gradients
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss}")

# Make predictions
y_pred_final = X_b.dot(weights)

# Plot the dataset and prediction
plt.scatter(X[:, 0], y, color='blue', label='Data')
plt.plot(X[:, 0], y_pred_final, color='red', label='Ridge Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge Regression Fit")
plt.legend()
plt.show()
