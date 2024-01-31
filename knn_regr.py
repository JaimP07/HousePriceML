import sklearn

import pandas as pd
from sklearn.datasets import fetch_california_housing
cal_housing_sk = fetch_california_housing(as_frame=True)
cal_housing = pd.DataFrame(cal_housing_sk.frame)
cal_housing.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

X = cal_housing.drop(columns=["MedHouseVal"])
Y = cal_housing["MedHouseVal"]

scaler = StandardScaler()
scaler.fit(X)
X_norm = pd.DataFrame(scaler.transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.2)

n_neighbors = 8
for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(x_train, y_train).predict(x_test)
    mae = mean_absolute_error(y_test, y_)
    print("KNN MAE: %.3f" % mae)
