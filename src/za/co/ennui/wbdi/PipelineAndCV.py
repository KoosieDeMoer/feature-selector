import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from ennuipy.FeatureSelector import FeatureSelector

# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

X = pd.read_excel (r'C:\Temp\learning\cleaner.xlsx')
y = pd.read_excel (r'C:\Temp\learning\targeter.xlsx') 


X.drop('Instance', axis=1, inplace=True)
y.drop('Instance', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regressor = BayesianRidge(normalize=True, n_iter=3, tol=0.001)
pipeline = make_pipeline(FeatureSelector(), regressor)


#pipeline.fit(X_train, y_train)

params = {'featureselector__feature_corr_min': [0.43, 0.44, 0.45],
          "featureselector__feature_x_corr_max": [0.31, 0.32, 0.33]}
grid = GridSearchCV(pipeline, param_grid=params, cv=4)

# DataConversionWarning: A column-vector y was passed when a 1d array was expected
y_train = y_train.squeeze()
y_test = y_test.squeeze()

grid.fit(X_train, y_train)
print(grid.best_params_)

#print(pipeline.named_steps['featureselector'].support_)

print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))

print(pipeline.get_params())
y_predict = grid.predict(X_test)
#print(pipeline.named_steps['bayesianridge'].coef_ )

plt.plot(y_test, y_predict, 'o');
plt.show()
