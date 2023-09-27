import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle
import warnings

cal=pd.read_csv('calories.csv')
# print(cal.head())
exer=pd.read_csv('exercise.csv')
# print(exer.head())

cal = pd.concat([exer, cal['Calories']], axis=1)
# print(cal.head())
# print(cal.isnull().sum())
# print(cal.describe())
cal.replace({"Gender":{'male':0,'female':1}}, inplace=True)
# print(cal.head())

x = cal.drop(columns=['User_ID','Calories'], axis=1)
y = cal['Calories']
# print(x)
# print(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(X_train)
print(Y_train)

model = XGBRegressor()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)

mae = metrics.mean_absolute_error(Y_test, y_pred)
# print(f"Mean Absolute Error : {mae}")

pickle.dump(model,open('burn.pkl','wb'))
burn=pickle.load(open('burn.pkl','rb'))
