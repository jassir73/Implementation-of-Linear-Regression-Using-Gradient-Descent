# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights randomly. 
2. Compute predicted values. 
3. Compute gradient of loss function.
4. Update weights using gradient descent

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: JASSIR SULTHAN K
RegisterNumber:  212224240060
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("/content/50_Startups.csv")
data.head()


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)


theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```


## Output:
![image](https://github.com/user-attachments/assets/e361f336-2baf-4dee-bb7c-8551c479f7c7)
![image](https://github.com/user-attachments/assets/ccfba393-91c0-4e65-a409-66aca5608ec6)
![image](https://github.com/user-attachments/assets/15dffb25-3d23-4c5b-b065-7063be0ec639)
![image](https://github.com/user-attachments/assets/01d22675-a235-4c97-8c34-249fcefa1fb2)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
