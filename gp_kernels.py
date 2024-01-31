import sklearn
import itertools
import numpy as np

import pandas as pd
from sklearn.datasets import fetch_california_housing
cal_housing_sk = fetch_california_housing(as_frame=True)
cal_housing = pd.DataFrame(cal_housing_sk.frame)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler

X = cal_housing.drop(columns=["MedHouseVal"])
Y = cal_housing["MedHouseVal"]

scaler = StandardScaler()
scaler.fit(X)
X_norm = pd.DataFrame(scaler.transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size=20000)

kernel1 = 1.0 * Matern(length_scale=[0.4]*8, nu=0.5)
kernel2 = 1.0 * RBF([0.4]*8)
kernel3 = 1.0 * RationalQuadratic(length_scale=0.4, alpha=1.5)
kernel4 = ExpSineSquared(length_scale=1.0, periodicity=0.01)
kernel5 = DotProduct() + WhiteKernel()

g_p1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=9)
g_p1.fit(x_train, y_train)
y_pred = g_p1.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Matern MAE: %.3f" % mae)

g_p2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9)
g_p2.fit(x_train, y_train)
y_pred = g_p2.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("RBF MAE: %.3f" % mae)

g_p3 = GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=9)
g_p3.fit(x_train, y_train)
y_pred = g_p3.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Rational Quadratic MAE: %.3f" % mae)

g_p4 = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=9)
g_p4.fit(x_train, y_train)
y_pred = g_p4.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Exp-Sine-Squared MAE: %.3f" % mae)

g_p5 = GaussianProcessRegressor(kernel=kernel5, n_restarts_optimizer=9)
g_p5.fit(x_train, y_train)
y_pred = g_p5.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Dot-Product MAE: %.3f" % mae)
