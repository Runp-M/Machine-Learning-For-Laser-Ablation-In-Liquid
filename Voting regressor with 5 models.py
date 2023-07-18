import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
%matplotlib inline

# Data Retrieval
df = pd.read_csv('your_data.csv')
df.head()
df.shape
df.info()

# creating features and label variable
X = df.drop(columns = ['target_variable'], axis = 1)
y = df['target_variable']
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# Checks if any of the characters '[', ']', '<' are present in the column name of train dataset and test dataset. If any of those characters are found, the expression regex.sub("_", col) replaces those characters with an underscore _.
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

# Validation dataset Retrieval
data = pd.read_csv('your_validation data.csv')
data.head()
data.shape

X1 = data.drop(columns = ['target_variable'], axis = 1)
y1 = data['target_variable']
X1.head()

# Checks if any of the characters '[', ']', '<' are present in the column name of validation dataset. If any of those characters are found, the expression regex.sub("_", col) replaces those characters with an underscore _.
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X1.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X1.columns.values]

# Load Xgboost model 

res = XGBRegressor(learning_rate = min(learning_rate, 0.99), max_depth = int(max_depth), n_estimators = int(n_estimators), gamma=min(gamma,0.99),
                        min_child_weight=int(min_child_weight),
                        subsample=min(subsample,2),
                        colsample_bytree=min(colsample_bytree,2),
                        reg_alpha=int(reg_alpha),
                        reg_lambda=int(reg_lambda),
                        seed=int(seed)
                        )
res.fit(X_train,y_train)

train_score = res.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = res.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = res.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred=res.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))


a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y1,ypred)
print('MAE:%f'%MAE)

# Loading Adaboost model

ada = AdaBoostRegressor(n_estimators = int(n_estimators), learning_rate = min(learning_rate, 2),  random_state = int( random_state),
                          base_estimator=res)
ada.fit(X_train, y_train)

train_score = ada.score(X_train, y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = ada.score(X_test, y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = ada.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred=ada.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Loading Gradient Boost

gbr = GradientBoostingRegressor( n_estimators =int(space['n_estimators']), learning_rate = space['learning_rate'], max_depth = int(space['max_depth']), max_features = str(space['max_features']),
                    loss=str(space['loss']), subsample=space['subsample'], random_state=int(space['random_state']))

gbr.fit(X_train, y_train)

train_score = gbr.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = gbr.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = gbr.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred = gbr.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))

# Loading Light Gradient Boost model

model_lgb = lgb.LGBMRegressor(learning_rate = min(learning_rate, 1.5), max_depth = int(max_depth), n_estimators = int(n_estimators),
                         min_child_samples=int(min_child_samples),
                        subsample=min(subsample,2),
                        colsample_bytree=min(colsample_bytree,2),
                        num_leaves = int(num_leaves),
                        min_split_gain = min(min_split_gain,2),
                        random_state = int(random_state),
                        reg_alpha=int(reg_alpha),
                        reg_lambda=int(reg_lambda))

model_lgb.fit(X_train, y_train)

train_score = model_lgb.score(X_train, y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = model_lgb.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = model_lgb.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred=model_lgb.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Catboost regressor

pip install catboost
from catboost import Pool, CatBoostRegressor

# specify the training parameters
model = CatBoostRegressor(iterations = int(iterations), learning_rate = min(learning_rate, 0.99), depth = int(depth), l2_leaf_reg=int(l2_leaf_reg),
                          random_strength = min(random_strength, 11),
                        loss_function='RMSE',
                        early_stopping_rounds=10,verbose=0)
#train the model
model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = model.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = model.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred=model.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Voting regressor

vtres = [ ('Ada Boost', ada1), ('XgBoost', res),('GradientBoost', gbr),('Light GradientBoost', model_lgb),('CatBoost', model)]
vtres = VotingRegressor(estimators = regressors, n_jobs = -1, verbose = 1, weights = (0.85,1,0.98,0.93,0.90))
vtres.fit(X_train, y_train)

train_score = vtres.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))
test_score = vtres.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))
Experimental_score = vtres.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred=vtres.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
