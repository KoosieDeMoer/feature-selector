import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.linear_model.sgd_fast import plain_sgd

home_advantage = 6

data = pd.read_excel (r'C:\Temp\learning\rugby results.xlsx')

X = data[['HOME', 'AWAY', 'HOME_ADVANTAGE', 'DATE_VALUE']]
y = data[['DIFF', 'HOME_ADVANTAGE']]

y = data[['DIFF', 'HOME_ADVANTAGE']]
y = y.apply(lambda row: row['DIFF'] if (row['HOME_ADVANTAGE'] == 'Y' ) else (row['DIFF'] - home_advantage), axis=1)

ohe = OneHotEncoder()

categories = np.array(list(set(X['HOME'].astype(str).values)) + list(set(X['AWAY'].astype(str).values))).reshape(-1,1)

ohe.fit(categories)

home = pd.DataFrame(ohe.transform(X[['HOME']]).toarray())
away = pd.DataFrame(ohe.transform(X[['AWAY']]).toarray())

X = pd.concat([home, away, X], axis=1)

X = X.drop(['HOME', 'AWAY', 'HOME_ADVANTAGE', 'DATE_VALUE'], axis=1)

X_train = X.iloc[298:591]
X_test = X.iloc[259:297]

y_train = y.iloc[298:591]
y_test = y.iloc[259:297]


#print(X_train)
#print(y_train)

regressor = MLPRegressor(hidden_layer_sizes=(20, 5), solver='sgd', max_iter=2000)

regressor.fit(X_train, y_train.squeeze().tolist())

print(regressor.score(X_train, y_train.squeeze().tolist()))
print(regressor.score(X_test, y_test.squeeze().tolist()))

print(regressor.get_params())
y_predict = regressor.predict(X_test)

plt.plot(y_test.squeeze().tolist(), y_predict, 'o');
plt.show()
