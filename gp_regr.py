import sklearn

import pandas as pd
from sklearn.datasets import fetch_california_housing
cal_housing_sk = fetch_california_housing(as_frame=True)
cal_housing = pd.DataFrame(cal_housing_sk.frame)
cal_housing.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

X = cal_housing.drop(columns=["MedHouseVal"])
Y = cal_housing["MedHouseVal"]

scaler = StandardScaler()
scaler.fit(X)
X_norm = pd.DataFrame(scaler.transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size=19640)

kernel = 1.0 * Matern(length_scale=[0.4]*8, nu=0.5)
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(x_train, y_train)

y_pred = gaussian_process.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("GP MAE: %.3f" % mae)
