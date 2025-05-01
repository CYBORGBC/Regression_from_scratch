import numpy as np
import pandas as pd

# 1. Load dataset
df = pd.read_csv('generated_dataset.csv')

# 2. Prepare X and y
X = df[['Age', 'Years_of_Education', 'Work_Experience']].values
y = df['Annual_Income'].values.reshape(-1, 1)  # Make it a column vector

# 3. Add a column of ones to X for bias (intercept)
X_b = np.hstack([X, np.ones((X.shape[0], 1))])  # Now last column is all 1s

# 4. Calculate weights using Normal Equation: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# 5. Predict
y_pred = X_b @ w

# 6. Evaluate
mse = np.mean((y - y_pred)**2)
print("Mean Squared Error:", mse)

# 7. Print coefficients
print("Coefficients:", w[:-1].flatten())
print("Intercept:", w[-1][0])
