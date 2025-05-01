import numpy as np
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
y_true = np.array([1, 0, 1, 1,  0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
print("Binary Cross-Entropy Loss:", binary_cross_entropy(y_true, y_pred))
