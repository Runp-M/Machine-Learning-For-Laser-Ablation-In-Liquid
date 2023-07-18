# The code will optimize each model separately using Bayesian Optimization and provide the best parameters and score for each model.
import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization

# Data Retrieval
df = pd.read_csv('your_data.csv')
X = df.drop(columns=['target_variable'], axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

def xgb_cv(learning_rate, max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, seed):
    res = XGBRegressor(learning_rate=min(learning_rate, 0.99), max_depth=int(max_depth), n_estimators=int(n_estimators), gamma=min(gamma, 0.99),
                       min_child_weight=int(min_child_weight), subsample=min(subsample, 2), colsample_bytree=min(colsample_bytree, 2),
                       reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda), seed=int(seed))
    res.fit(X_train, y_train)
    train_score = res.score(X_train, y_train)
    test_score = res.score(X_test, y_test)
    return test_score

def ada_cv(n_estimators, learning_rate, random_state):
    ada = AdaBoostRegressor(n_estimators=int(n_estimators), learning_rate=min(learning_rate, 2), random_state=int(random_state), base_estimator=res)
    ada.fit(X_train, y_train)
    train_score = ada.score(X_train, y_train)
    test_score = ada.score(X_test, y_test)
    return test_score

def gbr_cv(n_estimators, learning_rate, max_depth, max_features, loss, subsample, random_state):
    max_features_options = ['auto', 'sqrt', 'log2']
    loss_options = ['squared_error', 'absolute_error', 'huber', 'quantile']
    gbr = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth),
                                    max_features=max_features_options[int(max_features)], loss=loss_options[int(loss)],
                                    subsample=subsample, random_state=int(random_state))
    gbr.fit(X_train, y_train)
    train_score = gbr.score(X_train, y_train)
    test_score = gbr.score(X_test, y_test)
    return test_score

def cat_cv(iterations, learning_rate, depth, l2_leaf_reg, random_strength):
    cat = CatBoostRegressor(iterations=int(iterations), learning_rate=min(learning_rate, 0.99), depth=int(depth),
                            l2_leaf_reg=int(l2_leaf_reg), random_strength=min(random_strength, 11),
                            loss_function='RMSE', early_stopping_rounds=10, verbose=0)
    cat.fit(X_train, y_train)
    train_score = cat.score(X_train, y_train)
    test_score = cat.score(X_test, y_test)
    return test_score

def lgb_cv(learning_rate, max_depth, n_estimators, min_child_samples, subsample, colsample_bytree, num_leaves,
           min_split_gain, random_state, reg_alpha, reg_lambda):
    lgb = LGBMRegressor(learning_rate=min(learning_rate, 1.5), max_depth=int(max_depth), n_estimators=int(n_estimators),
                        min_child_samples=int(min_child_samples), subsample=min(subsample, 2),
                        colsample_bytree=min(colsample_bytree, 2), num_leaves=int(num_leaves),
                        min_split_gain=min(min_split_gain, 2), random_state=int(random_state),
                        reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda))
    lgb.fit(X_train, y_train)
    train_score = lgb.score(X_train, y_train)
    test_score = lgb.score(X_test, y_test)
    return test_score

pbounds = {'xgb': {'learning_rate': (0.001, 0.999),
                   'max_depth': (3, 30),
                   'n_estimators': (200, 1500),
                   'gamma': (0, 1),
                   'min_child_weight': (0, 10),
                   'subsample': (0.1, 1),
                   'colsample_bytree': (0.1, 1),
                   'reg_alpha': (0, 200),
                   'reg_lambda': (0, 200),
                   'seed': (0, 2023)},
           'ada': {'n_estimators': (200, 1000),
                   'learning_rate': (0.01, 1.2),
                   'random_state': (0, 200)},
           'gbr': {'n_estimators': (200, 900),
                   'learning_rate': (0.01, 0.5),
                   'max_depth': (3, 15),
                   'max_features': (0, 2),
                   'loss': (0, 3),
                   'subsample': (0, 0.6),
                   'random_state': (0, 100)},
           'cat': {'iterations': (200, 900),
                   'learning_rate': (0.01, 0.5),
                   'depth': (3, 15),
                   'l2_leaf_reg': (2, 10),
                   'random_strength': (0, 10)},
           'lgb': {'learning_rate': (0.001, 1),
                   'max_depth': (3, 100),
                   'n_estimators': (200, 1500),
                   'min_child_samples': (0, 12),
                   'subsample': (0.1, 1),
                   'colsample_bytree': (0.1, 1),
                   'num_leaves': (2, 100),
                   'min_split_gain': (0, 10),
                   'random_state': (0, 200),
                   'reg_alpha': (0, 200),
                   'reg_lambda': (0, 200)}}

models = {'xgb': xgb_cv, 'ada': ada_cv, 'gbr': gbr_cv, 'cat': cat_cv, 'lgb': lgb_cv}

def optimize_model(model_name):
    pbounds_model = pbounds[model_name]
    objective_model = models[model_name]
    optimizer = BayesianOptimization(f=objective_model, pbounds=pbounds_model)
    optimizer.maximize(init_points=10, n_iter=100)
    return optimizer.max

# Optimize XGBoost model
xgb_opt = optimize_model('xgb')
print("Optimized XGBoost Model:")
print("Best Score:", xgb_opt['target'])
print("Best Parameters:", xgb_opt['params'])

# Optimize AdaBoost model
ada_opt = optimize_model('ada')
print("Optimized AdaBoost Model:")
print("Best Score:", ada_opt['target'])
print("Best Parameters:", ada_opt['params'])

# Optimize Gradient Boosting model
gbr_opt = optimize_model('gbr')
print("Optimized Gradient Boosting Model:")
print("Best Score:", gbr_opt['target'])
print("Best Parameters:", gbr_opt['params'])

# Optimize CatBoost model
cat_opt = optimize_model('cat')
print("Optimized CatBoost Model:")
print("Best Score:", cat_opt['target'])
print("Best Parameters:", cat_opt['params'])

# Optimize LightGBM model
lgb_opt = optimize_model('lgb')
print("Optimized LightGBM Model:")
print("Best Score:", lgb_opt['target'])
print("Best Parameters:", lgb_opt['params'])
