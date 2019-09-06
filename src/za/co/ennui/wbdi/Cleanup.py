import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# minimum instances proportion to allow feature
minimum_valued_instances_proportion = 0.80

X = pd.read_excel (r'C:\Temp\learning\clean.xlsx')
y = pd.read_excel (r'C:\Temp\learning\happiness.xlsx') 

no_target_value_instances = y.notna().iloc[ : , 1 ]

X = X[ no_target_value_instances ]

y = y[ no_target_value_instances ]


X.dropna(thresh=len(X) * minimum_valued_instances_proportion, axis=1, inplace=True)

imp = IterativeImputer(max_iter=2, random_state=123)
start = time.time()

imp.fit(X.drop(X.columns[[0]], axis=1))

end = time.time()
print(end - start)

X_values = imp.transform(X.drop(X.columns[[0]], axis=1))


X.iloc[:,1:] = X_values

X.to_excel(r'C:\Temp\learning\cleaner.xlsx', index=False)
y.to_excel(r'C:\Temp\learning\targeter.xlsx', index=False)

print(X)
print(y)