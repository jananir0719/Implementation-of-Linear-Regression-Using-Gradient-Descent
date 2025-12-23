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
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Step 1: Create the dataset
# ------------------------------
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# ------------------------------
# Step 2: Split into X and Y
# ------------------------------
X = df[['Hours_Studied']]   # Feature (2D)
y = df['Marks_Scored']      # Target (1D)

# ------------------------------
# Step 3: Split data for training & testing
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Create and train the model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 6: Evaluate the model
# ------------------------------
print("\nModel Evaluation:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------------------
# Step 7: Visualize results
# ------------------------------
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()

# ------------------------------
# Step 8: Predict for new data
# ------------------------------
hours = float(input("\nEnter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for studying {hours} hours = {predicted_marks[0]:.2f}")

```

## Output:
<img width="532" height="488" alt="image" src="https://github.com/user-attachments/assets/8837bf20-38cc-4e1f-9008-19ff7d0099d8" />
<img width="971" height="737" alt="image" src="https://github.com/user-attachments/assets/25df48a4-2a47-4303-91c3-f39d87b522b8" />
<img width="730" height="67" alt="image" src="https://github.com/user-attachments/assets/ca80b5ba-f3ae-49df-a2bd-e12ced9d78d5" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
