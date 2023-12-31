# Title: Data-driven pre-determination of Cu oxidation state in copper nanoparticles: application to the synthesis by laser ablation in liquid
# Author: Runpeng Miao, Michael Bissoli, Andrea Basagni, Ester Marotta, Stefano Corni, Vincenzo Amendola,*
# Correspondence: runpeng.miao@phd.unipd.it, vincenzo.amendola@unipd.it

# This code utilizes a voting regressor to ensemble five tree-based models, each with optimized hyperparameters, in order to achieve the best predictive capability for the model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('fivethirtyeight')

# Data Retrieval
df = pd.read_csv('your_data.csv')
df.head()
df.shape
df.info()

# Create features and label variable
X = df.drop(columns=['target_variable'], axis=1)
y = df['target_variable']
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Checks if any of the characters '[', ']', '<' are present in the column name of the train dataset and test dataset. 
# If any of those characters are found, the expression regex.sub("_", col) replaces those characters with an underscore _.
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

# Validation dataset Retrieval
data = pd.read_csv('your_validation data.csv')
data.head()
data.shape

X1 = data.drop(columns=['target_variable'], axis=1)
y1 = data['target_variable']
X1.head()

# Checks if any of the characters '[', ']', '<' are present in the column name of the validation dataset. 
# If any of those characters are found, the expression regex.sub("_", col) replaces those characters with an underscore _.
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X1.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X1.columns.values]

# Load XGBoost model
res = XGBRegressor(learning_rate=min(learning_rate, 0.99), max_depth=int(max_depth), n_estimators=int(n_estimators), gamma=min(gamma, 0.99),
                   min_child_weight=int(min_child_weight), subsample=min(subsample, 2),
                   colsample_bytree=min(colsample_bytree, 2), reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda), seed=int(seed))
res.fit(X_train, y_train)

train_score = res.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = res.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = res.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = res.predict(X_train)
ypred_test = res.predict(X_test)
ypred = res.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

# Scatter plot of true values vs predictions
a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))

# Load AdaBoost model
ada = AdaBoostRegressor(n_estimators=int(n_estimators), learning_rate=min(learning_rate, 2), random_state=int(random_state),
                        base_estimator=res)
ada.fit(X_train, y_train)

train_score = ada.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = ada.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = ada.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = ada.predict(X_train)
ypred_test = ada.predict(X_test)
ypred = ada.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))

# Load Gradient Boost model
gbr = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth),
                                    max_features=max_features_options[int(max_features)], loss=loss_options[int(loss)],
                                    subsample=subsample, random_state=int(random_state))
gbr.fit(X_train, y_train)

train_score = gbr.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = gbr.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = gbr.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = gbr.predict(X_train)
ypred_test = gbr.predict(X_test)
ypred = gbr.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))

# Load Light Gradient Boost model
model_lgb = lgb.LGBMRegressor(learning_rate=min(learning_rate, 1.5), max_depth=int(max_depth), n_estimators=int(n_estimators),
                             min_child_samples=int(min_child_samples), subsample=min(subsample, 2),
                             colsample_bytree=min(colsample_bytree, 2), num_leaves=int(num_leaves),
                             min_split_gain=min(min_split_gain, 2), random_state=int(random_state),
                             reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda))
model_lgb.fit(X_train, y_train)

train_score = model_lgb.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = model_lgb.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = model_lgb.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = model_lgb.predict(X_train)
ypred_test = model_lgb.predict(X_test)
ypred = model_lgb.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))

# Catboost regressor
pip install catboost
from catboost import Pool, CatBoostRegressor

# Specify the training parameters
model = CatBoostRegressor(iterations=int(iterations), learning_rate=min(learning_rate, 0.99), depth=int(depth),
                          l2_leaf_reg=int(l2_leaf_reg), random_strength=min(random_strength, 11), loss_function='RMSE',
                          early_stopping_rounds=10, verbose=0)
# Train the model
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = model.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = model.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = model.predict(X_train)
ypred_test = model.predict(X_test)
ypred = model.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))

# Voting regressor
vtres = [('Ada Boost', ada), ('XgBoost', res), ('GradientBoost', gbr), ('Light GradientBoost', model_lgb), ('CatBoost', model)]
vtres = VotingRegressor(estimators=vtres, n_jobs=-1, verbose=1, weights=(0.85, 1, 0.98, 0.93, 0.90))
vtres.fit(X_train, y_train)

train_score = vtres.score(X_train, y_train)
print('Best score on train set: {:.2f}'.format(train_score))
test_score = vtres.score(X_test, y_test)
print('Best score on test set: {:.2f}'.format(test_score))
Experimental_score = vtres.score(X1, y1)
print('Best score on Experimental set: {:.2f}'.format(Experimental_score))

ypred_train = model.predict(X_train)
ypred_test = model.predict(X_test)
ypred = model.predict(X1)
print('r2_score: {:f}'.format(metrics.r2_score(y1, ypred)))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

MAE = mean_absolute_error(y_train, ypred_train)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_train, ypred_train,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y_test, ypred_test)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y_test, ypred_test,squared=False)
print('RMSE: {:f}'.format(RMSE))

MAE = mean_absolute_error(y1, ypred)
print('MAE: {:f}'.format(MAE))
RMSE = mean_squared_error(y1, ypred,squared=False)
print('RMSE: {:f}'.format(RMSE))
