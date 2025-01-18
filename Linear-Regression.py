import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
x_data_path = "C:/Users/Karan/OneDrive/Desktop/linearX.csv"
y_data_path = "C:/Users/Karan/OneDrive/Desktop/linearY.csv"

x_data = pd.read_csv(x_data_path, header=None).iloc[:, 0].values  # Independent variable
y_data = pd.read_csv(y_data_path, header=None).iloc[:, 0].values  # Dependent variable

# Normalize the data
x = (x_data - np.mean(x_data)) / np.std(x_data)  # Standardize x
y = (y_data - np.mean(y_data)) / np.std(y_data)  # Standardize y

# Add a column of ones to x for the bias term (intercept)
x = np.c_[np.ones(len(x)), x]

# Batch Gradient Descent
def batch_gradient_descent(x, y, learning_rate, max_iter, tolerance):
    m, n = x.shape
    theta = np.zeros(n)
    cost_history = []

    for iteration in range(max_iter):
        predictions = np.dot(x, theta)
        errors = predictions - y

        # Compute the cost (MSE)
        cost = np.mean(errors ** 2) / 2
        cost_history.append(cost)

        # Compute the gradient
        gradients = np.dot(x.T, errors) / m

        # Update the parameters
        theta -= learning_rate * gradients

        # Check for convergence
        if np.linalg.norm(gradients) < tolerance:
            break

    return theta, cost_history

# Train the model using Batch Gradient Descent
learning_rate = 0.5
max_iter = 50
tolerance = 1e-6
theta_batch, cost_history_batch = batch_gradient_descent(x, y, learning_rate, max_iter, tolerance)

# Plot 1: Data Points
plt.figure(figsize=(8, 5))
plt.scatter(x[:, 1], y, color='blue', label="Data points")
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Data Points")
plt.legend()
plt.show()

# Plot 2: Linear Regression with Gradient Descent
plt.figure(figsize=(8, 5))
plt.scatter(x[:, 1], y, color='blue', label="Data points")
plt.plot(x[:, 1], np.dot(x, theta_batch), color='red', label="Regression line")
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()

# Plot 3: Cost Function vs Iterations (First 50 Iterations)
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history_batch)), cost_history_batch, label="Cost function", color='green')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function vs Iterations (First 50 Iterations)")
plt.legend()
plt.show()

# Plot 4: Cost Function Convergence
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history_batch)), cost_history_batch, label="Cost function convergence", color='purple')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.legend()
plt.show()

# Print Results
print("Theta (Batch GD):", theta_batch)
print("Final cost (Batch GD):", cost_history_batch[-1])