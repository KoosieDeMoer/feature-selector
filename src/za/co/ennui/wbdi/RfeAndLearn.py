import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

X = pd.read_excel (r'C:\Temp\learning\cleaner.xlsx')
y = pd.read_excel (r'C:\Temp\learning\targeter.xlsx') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.drop('Instance', axis=1, inplace=True)
X_test.drop('Instance', axis=1, inplace=True)





# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)

y_train = y_train[['Target']]

y_test = y_test[['Target']]


regressor = BayesianRidge(normalize=True, n_iter=5, tol=0.01)
# regressor = LinearRegression()
pipeline = make_pipeline(RFE(regressor, 4), regressor)


pipeline.fit(X_train, y_train.squeeze().tolist())

selected_features = []
for i in range(len(X_train.columns)): 
    if(pipeline.named_steps['rfe'].support_[i]):
        selected_features.append(X_train.columns[i]) 

print(selected_features)

print(pipeline.score(X_train, y_train.squeeze().tolist()))
print(pipeline.score(X_test, y_test.squeeze().tolist()))

print(pipeline.get_params())
y_predict = pipeline.predict(X_test)
print(pipeline.named_steps['bayesianridge'].coef_ )

plt.plot(y_test.squeeze().tolist(), y_predict, 'o');
plt.show()
