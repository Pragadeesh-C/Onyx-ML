import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


df=pd.read_csv('Diet.csv')
df.replace({"gender":{'M':0,'F':1}}, inplace=True)
x=df.iloc[:,1:8]
print(x)
y=df.iloc[:,8]
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# X_train_encoded = pd.get_dummies(X_train, columns=['gender'], drop_first=True)
# X_test_encoded = pd.get_dummies(X_test, columns=['gender'], drop_first=True)

# Create and train an XGBoost regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = SVR(kernel='linear')
model.fit(X_train, y_train)
print(X_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
pickle.dump(model,open('Diet.pkl','wb'))
burn=pickle.load(open('Diet.pkl','rb'))