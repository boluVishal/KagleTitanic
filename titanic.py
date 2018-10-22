# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
# Importing the dataset
dataset = pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
X = dataset.iloc[0:,[2,4,5,6,7,9,11]].values
y = dataset.iloc[:, 1].values
X_test= testData.iloc[0:,[1,3,4,5,6,8,10]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[0:, 2:3])
X[0:, 2:3] = imputer.transform(X[0:, 2:3])

imputer_t = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_t = imputer_t.fit(X_test[0:, 2:3])
X_test[0:, 2:3] = imputer_t.transform(X_test[0:, 2:3])
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])
X[:, -1] = labelencoder_X.fit_transform(X[:,-1])
labelencoder_Xtest = LabelEncoder()
X_test[:, 1] = labelencoder_Xtest.fit_transform(X_test[:,1])
X_test[:, -1] = labelencoder_Xtest.fit_transform(X_test[:,-1])

onehotencoder = OneHotEncoder(categorical_features = [-1])
onehotencoder1 = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()
X_test= onehotencoder1.fit_transform(X_test).toarray()
X=X[:,1:]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X)

# Encoding the Dependent Variable
