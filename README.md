# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Intialize weights randomly. 2.Compute predicted. 3.Compute gradient of loss function. 4.Update weights using gradient descent.

## Program:
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
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")
```

## Output:
![Machine Learning 1](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/7fcbc1d1-43dc-43ed-ae21-6868878c6ef8)

![2](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/5bd92552-d361-41cf-8b3a-4cba9f359aca)

![3](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/c4ed46dd-719e-4205-848f-f04bb6cdb1f4)

![4](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/10c87e67-b003-468a-b739-e7d0a08d0d93)
![5](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/aca29fc8-5bfd-424b-93a5-5c236a3b2b59)

![6](https://github.com/RamkumarGunasekaran/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870820/2df83165-0ea3-4446-be9b-a028c075f254)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
