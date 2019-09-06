import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
# from sklearn.linear_model import ARDRegression
# from sklearn.linear_model import SGDRegressor
# from sklearn.neural_network.multilayer_perceptron import MLPRegressor
# from sklearn.linear_model import RANSACRegressor
# from sklearn.linear_model import Lasso

X = pd.read_excel (r'C:\Temp\learning\cleaner.xlsx')
y = pd.read_excel (r'C:\Temp\learning\targeter.xlsx') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.drop('Instance', axis=1, inplace=True)
X_test.drop('Instance', axis=1, inplace=True)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)

print (y_train)
y_train = y_train[['Target']]

print (y_train)
y_test = y_test[['Target']]


# corr = data.corr()

# param_grid = {'C': [4.7, 4.8, 4.9, 5.0], 'gamma': [ 0.000009, 0.000010, 0.000011, 0.000012]}


print(X_train)
print(y_train)

# regressor = LinearRegression()
# regressor = SVR(C=5, gamma=0.00001)
regressor = BayesianRidge(normalize=True, n_iter=5, tol=0.01, fit_intercept=True)
# regressor = ARDRegression(normalize=True, n_iter=5, tol=0.01)
# regressor = SGDRegressor()
# regressor = MLPRegressor(hidden_layer_sizes=(200, 50, 10))
# regressor = RANSACRegressor(min_samples=80, max_trials=1000)
# regressor = Lasso()

regressor.fit(X_train, y_train.squeeze().tolist())

print(regressor.score(X_train, y_train.squeeze().tolist()))
print(regressor.score(X_test, y_test.squeeze().tolist()))

print(regressor.get_params())
y_predict = regressor.predict(X_test)
print(y_predict)

plt.plot(y_test.squeeze().tolist(), y_predict, 'o');
plt.show()
