import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel (r'C:\Temp\learning\rugby results.xlsx')

X = data[['HOME', 'AWAY', 'HOME_ADVANTAGE', 'DATE_VALUE']]
y = data[['DIFF']]

le = OrdinalEncoder()
le.fit(np.array(list(set(X['HOME'].astype(str).values)) + list(set(X['AWAY'].astype(str).values))).reshape(-1,1))
print(le.categories_)


home = pd.DataFrame(le.transform(X[['HOME']]))
away = pd.DataFrame(le.transform(X[['AWAY']]))

X = pd.concat([home, away, X], axis=1)

X = X.drop(['HOME', 'AWAY', 'DATE_VALUE'], axis=1)

X_train = X.iloc[298:591]
X_test = X.iloc[259:297]

y_train = y.iloc[298:591]
y_test = y.iloc[259:297]


#print(X_train)
#print(y_train)

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(X_train, y_train.squeeze().tolist())

print(regressor.score(X_train, y_train.squeeze().tolist()))
print(regressor.score(X_test, y_test.squeeze().tolist()))

print(regressor.get_params())
y_predict = regressor.predict(X_test)

plt.plot(y_test.squeeze().tolist(), y_predict, 'o');
plt.show()
