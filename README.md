# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the CSV file, separate features and target, convert to NumPy arrays, and scale them using StandardScaler.

2.Feature Engineering: Add a column of ones to the feature matrix to include the bias term in the model.

3.Initialization: Initialize weight vector theta with zeros, including one for each feature and the bias.

4.Gradient Descent: Loop for a fixed number of iterations, compute predictions, calculate errors, and update weights using the gradient formula.

5.Prediction: Standardize new input using the same scaler and compute the predicted output using the learned weights.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: JASSIR SULTHAN K
RegisterNumber:  212224240060

```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear regression using gradient descent
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]  # Add intercept term
    theta = np.zeros((X.shape[1], 1))  # Initialize theta

    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta

# Load dataset (with header)
data = pd.read_csv('50_Startups.csv')

# Use first 3 columns (R&D Spend, Administration, Marketing Spend) as features
X = data.iloc[:, 0:3].values.astype(float)

# Use the last column (Profit) as target
y = data.iloc[:, -1].values.reshape(-1, 1)

# Standardize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train the model
theta = linear_regression(X_scaled, y_scaled)

# Predict for a new data point
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = scaler_X.transform(new_data)
new_scaled_with_bias = np.c_[np.ones((new_scaled.shape[0], 1)), new_scaled]
prediction_scaled = new_scaled_with_bias.dot(theta)
prediction = scaler_y.inverse_transform(prediction_scaled)

# Print predicted value
print(f"Predicted Profit: {prediction}")
```

## Output:
![image](https://github.com/user-attachments/assets/02d2debe-63a8-4f83-b604-a6cde4d3653b)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
