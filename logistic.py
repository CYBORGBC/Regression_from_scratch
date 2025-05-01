import numpy as np
import pandas as pd

# 1. Load dataset
df = pd.read_csv('generated_dataset.csv')

# 2. Prepare X and y
X = df[['Age', 'Years_of_Education', 'Work_Experience', 'Annual_Income']].values
y = df['Owns_Car'].values.reshape(-1, 1)

# 3. Initialize parameters
n_samples, n_features = X.shape
weights = np.zeros((n_features, 1))
bias = 0
learning_rate = 0.001
n_iterations = 10000

# 4. Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 5. Gradient Descent
for i in range(n_iterations):
    # Forward pass
    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)

    # Compute gradients
    dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
    db = (1 / n_samples) * np.sum(y_predicted - y)

    # Update parameters
    weights -= learning_rate * dw
    bias -= learning_rate * db

    # Optional: Print loss every 1000 steps
    if i % 1000 == 0:
        loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
        print(f"Iteration {i}: Loss = {loss}")

# 6. Predict labels
y_pred_labels = (sigmoid(np.dot(X, weights) + bias) >= 0.5).astype(int)

# 7. Accuracy
accuracy = np.mean(y_pred_labels == y)
print("Accuracy:", accuracy)

# 8. Print weights and bias
print("Weights:", weights.flatten())
print("Bias:", bias)
