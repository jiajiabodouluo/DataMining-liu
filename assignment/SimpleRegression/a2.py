import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.2,random_state=0)

# feature scaling
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

# predicting the test set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Median home price vs interest rate')
plt.xlabel('interest rate')
plt.ylabel('Median home price')
plt.show()
