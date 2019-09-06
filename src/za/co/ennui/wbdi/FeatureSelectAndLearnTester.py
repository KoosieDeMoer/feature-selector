import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import BayesianRidge
from ennuipy.FeatureSelector import FeatureSelector


X = pd.read_excel (r'C:\Temp\learning\cleaner.xlsx')
y = pd.read_excel (r'C:\Temp\learning\targeter.xlsx') 

X.drop('Instance', axis=1, inplace=True)
y.drop('Instance', axis=1, inplace=True)
 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


feature_selector = FeatureSelector(feature_corr_min=0.45, feature_x_corr_max=0.5)

feature_selector.fit(X_train, y_train)

X_train = feature_selector.transform(X_train)
X_test = feature_selector.transform(X_test)



# corr = data.corr()

# param_grid = {'C': [4.7, 4.8, 4.9, 5.0], 'gamma': [ 0.000009, 0.000010, 0.000011, 0.000012]}


#print(X_train)
#print(y_train)

# regressor = LinearRegression()
regressor = BayesianRidge()
#regressor.fit(X_train, y_train.squeeze().tolist())
regressor.fit(X_train, y_train)

print('Fit=' + str(regressor.score(X_train, y_train)))
print('Score=' + str(regressor.score(X_test, y_test)))
print(feature_selector.features_)

print(regressor.get_params())
y_predict = regressor.predict(X_test)

plt.plot(y_test, y_predict, 'o');
plt.show()
