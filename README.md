# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters
2. compute predictions and cost
3. Update parameters using gradent descent
4. Predict output

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JANANI R
RegisterNumber:  25018734
*/
import numpy as np
import matplotlib.pyplot as plt

# Sample training data (Population of City, Profit)
X = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829])
y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233, 13.662])

# Number of samples
m = len(y)

# Add column of 1s for bias term
X_b = np.c_[np.ones((m, 1)), X]

# Initialize parameters
theta = np.zeros(2)

# Gradient Descent settings
alpha = 0.01
iterations = 1500

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Gradient descent algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta = theta - alpha * gradient
    return theta

# Train model
theta = gradient_descent(X_b, y, theta, alpha, iterations)

print("Theta values:", theta)

# Predict profit for any city population
population = float(input("Enter city population: "))
prediction = theta[0] + theta[1] * population
print("Predicted Profit:", prediction)

```

## Output:
<img width="492" height="116" alt="image" src="https://github.com/user-attachments/assets/b297d001-a771-460d-99ef-6b7aa38b803f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
