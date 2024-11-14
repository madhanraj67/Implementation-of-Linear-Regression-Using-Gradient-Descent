# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries as Load numpy, pandas, and StandardScaler for data processing.
2. Add a bias term to features (X).Initialize theta (coefficients) to zeros.Update theta using Gradient Descent
3. Read dataset (50_Startups.csv).Extract features (X) and target (y) from the dataset.
4. Normalize X and y using StandardScaler for better gradient descent performance.
5. Pass scaled X and y to the linear_regression() function to compute theta.## Program:
6. Use the model equation:𝑦pred=[1,𝑋scaled]⋅𝜃y pred​ =[1,X scaled​ ]⋅θ
7. Print the final predicted output.
```
/*
Program to implement the linear regression using gradient descent.
Developed by:P.MADHANRAJ
RegisterNumber:212223220052
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv');
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([16539.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
```
# Output:
#![Screenshot 2024-08-30 112656](https://github.com/user-attachments/assets/1072069c-37a1-4dc7-9873-b8718f3faf39)
![Screenshot 2024-08-30 112708](https://github.com/user-attachments/assets/5b900e04-53ae-4476-8187-f423aacb8c51)

![Screenshot 2024-08-30 112715](https://github.com/user-attachments/assets/2e70a6e0-9e41-4164-b391-54733b3bc586)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
