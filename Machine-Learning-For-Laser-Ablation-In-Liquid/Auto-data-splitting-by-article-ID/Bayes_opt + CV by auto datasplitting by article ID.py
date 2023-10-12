# Title: Data-driven pre-determination of Cu oxidation state in copper nanoparticles: application to the synthesis by laser ablation in liquid
# Author: Runpeng Miao, Michael Bissoli, Andrea Basagni, Ester Marotta, Stefano Corni, Vincenzo Amendola,*
# Correspondence: runpeng.miao@phd.unipd.it, vincenzo.amendola@unipd.it

# The code employs Bayesian optimization to optimize the hyperparameters of the XGBoost model on diverse datasets obtained through the 'extract_rows' function of the 'Auto data splitting by article ID.py'. 
# The primary objective is to achieve the highest average R2 score across a range of datasets from various article sources.
# This optimization can be applied to any machine learning models on any datasets from different article sources.

def res_cv(learning_rate, max_depth, n_estimators,gamma,min_child_weight,subsample,colsample_bytree,reg_alpha,
          reg_lambda,seed):
    r2_scores = []  # List to store average R2 scores

    for _ in range(20):  # Run 20 iterations with different random numbers
        data = pd.read_excel('your_data.xlsx')
        print("data shape:", data.shape)
      
        X_train, y_train, X_test, y_test, df_80_percent, df_20_percent,random_number= extract_rows(data)
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        res = XGBRegressor(
            learning_rate = min(learning_rate, 0.99), max_depth = int(max_depth), n_estimators = int(n_estimators), gamma=min(gamma,0.99),
                        min_child_weight=int(min_child_weight),
                        subsample=min(subsample,2),
                        colsample_bytree=min(colsample_bytree,2),
                        reg_alpha=int(reg_alpha),
                        reg_lambda=int(reg_lambda),
                        seed=int(seed)
        )

        res.fit(X_train, y_train)
        y_pred = res.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        print("r2_scores ", r2_scores)
      
        # Record all of r2 score appearing during all of iterations and get the best 1 and 2 score
        all_r2_scores.append(r2_score(y_test, y_pred))  # Append the R2 score to the all_r2_scores list
        all_random_numbers.append(random_number)
        all_hyperparameters.append({
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'seed': seed
          })
              
    avg_r2_score = np.mean(r2_scores)
    r2_scores.clear()
    print("avg_r2_score ", avg_r2_score)

    return avg_r2_score # Return the average of the average R2 scores

# Example usage
from bayes_opt import BayesianOptimization

# Define the parameter bounds for optimization
param_bounds = {
        'learning_rate': (0.001,0.999),
        'max_depth': (3,30),
        'n_estimators': (200,1500),
         'gamma': (0,1),
         'min_child_weight': (0,10),
         'subsample': (0.1,1),
         'colsample_bytree': (0.1,1),
         'reg_alpha': (0,200),
         'reg_lambda': (0,200),
         'seed':(0,2023)
}

# Run Bayesian optimization
optimizer = BayesianOptimization(
    res_cv,
    pbounds=param_bounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(init_points=15,n_iter=1000)

# Get the best hyperparameters and the corresponding R2 score
best_params = optimizer.max['params']
best_score = optimizer.max['target']

print("Best Hyperparameters:", best_params)
print("Best R2 Score:", best_score)
