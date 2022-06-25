# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Start the  program
2. import numpy,pandas,matplotlib
3. import the required csv file
4. print data.head(),data.tail()
5. Assign the values for x & y
6. import train_test_split and split the dataset into test data and train data
7. import LinearRegression and plot the graph using matplotlib libraries
8. Stop the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S Saichandran 
RegisterNumber:  212220040138
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("/content/student_scores.csv")
dataset.head() 
dataset.tail()  
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,1].values   
print(X)
print(y)
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title("h vs s(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("h vs s(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](/1.%20data.head().PNG)
![simple linear regression model for predicting the marks scored](/2.%20data.tail().PNG)
![simple linear regression model for predicting the marks scored](/x%20%26%20y.PNG)
![simple linear regression model for predicting the marks scored](/output%201.PNG)
![simple linear regression model for predicting the marks scored](/output%202.PNG)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
