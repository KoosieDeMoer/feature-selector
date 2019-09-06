import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import BayesianRidge

#from sklearn.svm import SVR
# any feature with less than this level of correlation to the target will be ignored
feature_correlation_threshold = 0.45

# any features that have a x_correlation with a selected feature higher than this will be eliminated
feature_x_correlation_threshold = 0.5

X = pd.read_excel (r'C:\Temp\learning\cleaner.xlsx')
y = pd.read_excel (r'C:\Temp\learning\targeter.xlsx') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data = pd.concat([y_train, X_train], axis=1, sort=False)


corr = data.corr()



# these are the features to do some model building with
features = []
result = corr.reindex(corr['Target'].abs().sort_values( ascending=False ).index)
result.drop('Target', inplace=True)

while not result.empty:
    if ((result[result['Target'].abs() > feature_correlation_threshold]).empty):
        break
    feature_name = result.index.values[0]
    features.append(feature_name)
    result.drop(feature_name, inplace=True)
    correlated_features = []
    result = result[result[feature_name].abs() < feature_x_correlation_threshold]




print (features)


X_train = X_train[features]
X_test = X_test[features]

print (y_train)
y_train = y_train[['Target']]

print (y_train)
y_test = y_test[['Target']]


# corr = data.corr()

# param_grid = {'C': [4.7, 4.8, 4.9, 5.0], 'gamma': [ 0.000009, 0.000010, 0.000011, 0.000012]}


print(X_train)
print(y_train)

# regressor = LinearRegression()
regressor = BayesianRidge()
#regressor.fit(X_train, y_train.squeeze().tolist())
regressor.fit(X_train, y_train.squeeze().tolist())

print(regressor.score(X_test, y_test.squeeze().tolist()))

print(regressor.get_params())
y_predict = regressor.predict(X_test)
print(y_predict)

plt.plot(y_test.squeeze().tolist(), y_predict, 'o');
plt.show()
