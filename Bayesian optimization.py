# The code will optimize each model separately using Bayesian Optimization and provide the best parameters and score for each model.

pip install catboost
pip install bayesian-optimization

import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Data Retrieval
df = pd.read_csv('your_data.csv')
X = df.drop(columns=['target_variable'], axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

def optimize_xgb(learning_rate, max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, seed):
    xgb = XGBRegressor(learning_rate=min(learning_rate, 0.99), max_depth=int(max_depth), n_estimators=int(n_estimators), gamma=min(gamma, 0.99),
                       min_child_weight=int(min_child_weight), subsample=min(subsample, 2), colsample_bytree=min(colsample_bytree, 2),
                       reg_alpha=int(reg_alpha), reg_lambda=int(reg_lambda), seed=int(seed))
    xgb.fit(X_train, y_train)
    train_score = xgb.score(X_train, y_train)
    test_score = xgb.score(X_test, y_test)
    return test_score

def optimize_ada(n_estimators, learning_rate, random_state):
    ada = AdaBoostRegressor(n_estimators=int(n_estimators), learning_rate=min(learning_rate, 2), random_state=int(random_state), base_estimator=res)
    ada.fit(X_train, y_train)
    train_score = ada.score(X_train, y_train)
    test_score = ada.score(X_test, y_test)
    return test_score

def optimize_gbr(n_estimators, learning_rate, max_depth, max_features, loss, subsample, random_state):
    max_features_options = ['auto', 'sqrt', 'log2']
    loss_options = ['squared_error', 'absolute_error', 'huber', 'quantile']
    gbr = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth),
                                    max_features=max_features_options[int(max_features)], loss=loss_options[int(loss)],
                                    subsample=subsample, random_state=int(random_state))
    gbr.fit(X_train, y_train)
    train_score = gbr.score(X_train, y_train)
    test_score = gbr.score(X_test, y_test)
    return test_score

def optimize_cat(iterations, learning_rate, depth, l2_leaf_reg, random_strength):
    cat = CatBoostRegressor(iterations=int(iterations), learning_rate=min(learning_rate, 0.99), depth=int(depth),
                            l2_leaf_reg=int(l2_leaf_reg), random_strength=min(random_strength, 11),
                            loss_function='RMSE', early_stopping_rounds=10, verbose=0)
    cat.fit(X_train, y_train)
    train_score = cat.score(X_train, y_train)
    test_score = cat.score(X_test, y_test)
    return test_score

def optimize_lgb(learning_rate, max_depth, n_estimators, min_child_samples, subsample, colsample_bytree, num_leaves,
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

def optimize_mlp(hidden_layer_sizes_1, hidden_layer_sizes_2, alpha, learning_rate_init, max_iter, tol, momentum,
                 validation_fraction, random_state, beta_1, beta_2, epsilon,
                 n_iter_no_change, max_fun):
 
    # Setting hidden layer sizes as a iterable parameter in Bayesian optimization function. The quantity of element in hidden_layer_sizes represent the quantity of hidden layer in neural network                 
    hidden_layer_sizes = (int(hidden_layer_sizes_1), int(hidden_layer_sizes_2),)
    MLP = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        alpha=alpha,
                        learning_rate_init=learning_rate_init,
                        max_iter=int(max_iter),
                        tol=tol,
                        momentum=momentum,
                        validation_fraction=validation_fraction,
                        random_state=int(random_state),
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        n_iter_no_change=int(n_iter_no_change),
                        max_fun=int(max_fun))
    normalize = MinMaxScaler()
    normalized_X_train = normalize.fit_transform(X_train)
    normalized_X_test = normalize.fit_transform(X_test)
    normalized_X1 = normalize.fit_transform(X1)
                     
    MLP.fit(normalized_X_train, y_train)
    train_score = MLP.score(normalized_X_train, y_train)
    test_score = MLP.score(normalized_X_test, y_test)

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
                   'reg_lambda': (0, 200)}
           'MLP': {'hidden_layer_sizes_1': (1, 100),
                   'hidden_layer_sizes_2': (1, 100),
                   'alpha': (0.0001, 0.1),
                   'learning_rate_init': (0.0001, 0.1),
                   'max_iter': (100, 1000),
                   'tol': (0.0001, 0.01),
                   'momentum': (0.1, 0.9),
                   'validation_fraction': (0.1, 0.3),
                   'random_state' :(1,5000),
                   'beta_1': (0.8, 0.99),
                   'beta_2': (0.99, 0.999),
                   'epsilon': (1e-08, 1e-06),
                   'n_iter_no_change': (5, 20),
                   'max_fun': (10000, 20000)
          }}

models = {'xgb': optimize_xgb, 'ada': optimize_ada, 'gbr': optimize_gbr, 'cat': optimize_cat, 'lgb': optimize_lgb, 'MLP':optimize_mlp}

def optimize_model(model_name):
    pbounds_model = pbounds[model_name]
    objective_model = models[model_name]
    optimizer = BayesianOptimization(f=objective_model, pbounds=pbounds_model)
    optimizer.maximize(init_points=10, n_iter=1000)
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

# Optimize Neural network model
MLP_opt = optimize_model('MLP')
print("Optimized LightGBM Model:")
print("Best Score:", MLP_opt['target'])
print("Best Parameters:", MLP_opt['params'])
