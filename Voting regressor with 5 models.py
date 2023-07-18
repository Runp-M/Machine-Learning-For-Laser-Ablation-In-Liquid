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
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
%matplotlib inline

# Import data file
df = pd.read_csv('Data for ML correct after ranking 07-5-2023.csv')
df.head()
df.shape
df.info()

# creating features and label variable

X = df.drop(columns = ['Output 1: Mean oxidation state','boiling point of solvent (K)','refractive index at 589 nm of solvent'], axis = 1)
y = df['Output 1: Mean oxidation state']
# X = X.drop(columns = '# pulses', axis = 1)

X = pd.get_dummies(X, drop_first = True)
X.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# # scaling data


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

data = pd.read_csv('experimental dataset 07-05-2023.csv')
data.head()

data.shape

X1 = data.drop(columns = ['Output 1: Mean oxidation state','boiling point of solvent (K)','refractive index at 589 nm of solvent'], axis = 1)
y1 = data['Output 1: Mean oxidation state']
# X1 = X1.drop(columns = '# pulses', axis = 1)

X1 = pd.get_dummies(X1, drop_first = True)
X1.head()

y1

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X1.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X1.columns.values]

## Xgboost

res = XGBRegressor(booster = 'gbtree', learning_rate = 0.999, max_depth = 3, n_estimators =962, objective='reg:squarederror', gamma=0,
                        min_child_weight=0,
                        subsample=1,
                        colsample_bytree=0.4844,
                        reg_alpha=0,
                        reg_lambda=55,
                        seed=1084,
                        scale_pos_weight=1,
                        max_delta_step=0,
                        colsample_bylevel=1
                        )
res.fit(X_train,y_train)

# res = XGBRegressor(booster = 'gbtree', learning_rate = 0.9926509145222427, max_depth = 5, n_estimators =1230, objective='reg:squarederror', gamma=0.05213343475881005,
#                         min_child_weight=0,
#                         subsample=0.6989658105516487,
#                         colsample_bytree= 0.38216551006513666,
#                         reg_alpha=1,
#                         reg_lambda=7,
#                         seed=859,
#                         scale_pos_weight=1,
#                         max_delta_step=0,
#                         colsample_bylevel=1
#                         )
# res.fit(X_train,y_train)

test_score = res.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))

train_score = res.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))

Experimental_score = res.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred1=res.predict(X1)
ypred1 = res.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred1))

y1

ypred1

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred1)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y1,ypred1)
print('MAE:%f'%MAE)

### **Adaboost**

ada1 = AdaBoostRegressor(base_estimator=res, n_estimators=781, learning_rate=0.019540128110512693, loss='linear', random_state=186)
ada1.fit(X_train, y_train)

ada1.score(X_train, y_train)

ada1.score(X_test, y_test)

Experimental_score = ada1.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred1=ada1.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred1))

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred1)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

## **Gradient Boost**

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=690, learning_rate=0.05708351622363626,max_depth=9,max_features='auto',loss='squared_error',subsample= 0.3576765627770961, random_state=29)
gbr.fit(X_train, y_train)

gbr.score(X_train, y_train)

gbr.score(X_test, y_test)

Experimental_score = gbr.score(X1,y1)
print('Best score on test set:{:.2f}'.format(Experimental_score))

## **Light Gradient Boost**

model_lgb = lgb.LGBMRegressor(learning_rate = 0.702616373421449, max_depth =  27, n_estimators = 1134,
                        min_child_samples=6,
                        subsample=0.8374480357275413,
                        colsample_bytree=0.1966605440352757,
                        num_leaves = 58,
                        min_split_gain = 0,
                        random_state = 31,
                        reg_alpha=0,
                        reg_lambda=129)
model_lgb.fit(X_train, y_train)

test_score = model_lgb.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))

train_score = model_lgb.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))

Experimental_score = model_lgb.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

### **Catboost regressor**

pip install catboost

from catboost import Pool, CatBoostRegressor

# specify the training parameters
model = CatBoostRegressor(iterations=485,
                          depth=8,
                          learning_rate=0.44614642251058245,
                          l2_leaf_reg = 8,
                          random_strength =3.2441002701899833,
                          loss_function='RMSE',early_stopping_rounds=10)
#train the model
model.fit(X_train,y_train)

X_train.shape

test_score = model.score(X_test,y_test)
print('Best score on test set:{:.2f}'.format(test_score))

train_score = model.score(X_train,y_train)
print('Best score on train set:{:.2f}'.format(train_score))

Experimental_score = model.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

## **Voting regressor**

from sklearn.ensemble import VotingRegressor

regressors = [ ('Ada Boost', ada1), ('XgBoost', res),('GradientBoost', gbr),('Light GradientBoost', model_lgb),('CatBoost', model)]

vr = VotingRegressor(estimators = regressors, n_jobs = -1, verbose = 1, weights = (0.85,1,0.98,0.93,0.90))
vr.fit(X_train, y_train)

(0.6, 0.9,0.8)

vr.score(X_train, y_train)

vr.score(X_test, y_test)

Experimental_score = vr.score(X1,y1)
print('Best score on Experimental set:{:.2f}'.format(Experimental_score))

ypred = vr.predict(X1)
ypred

ypred = pd.DataFrame(ypred, columns=['Prediction value'])
ypred

X_test

pd.DataFrame(np.array(y_test), columns=['real value'])

np.array(y_test)

ypred2 = vr.predict(X_train)

ypred1=vr.predict(X1)
print('r2_score:%f'%metrics.r2_score(y1,ypred1))

ypred1

y1

a = plt.axes(aspect='equal')
plt.scatter(y1, ypred1)
plt.xlabel('True Values [Output1]')
plt.ylabel('Predictions [Output1]')
lims = [-1, 3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
