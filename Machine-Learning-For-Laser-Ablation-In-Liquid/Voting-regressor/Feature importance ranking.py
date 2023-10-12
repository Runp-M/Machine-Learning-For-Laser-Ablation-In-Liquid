# Title: Data-driven pre-determination of Cu oxidation state in copper nanoparticles: application to the synthesis by laser ablation in liquid
# Author: Runpeng Miao, Michael Bissoli, Andrea Basagni, Ester Marotta, Stefano Corni, Vincenzo Amendola,*
# Correspondence: runpeng.miao@phd.unipd.it, vincenzo.amendola@unipd.it

# This code implements the calculation of feature importance of all features by using the dataset of interest fitted in Decision Tree model. 
# This code could be applied to any ML models basde on sklearn library.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Data Retrieval
df = pd.read_csv('your_data.csv')
X = df.drop(columns=['target_variable'], axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)

# Determination of feature importance
features = X.columns # obtain feature name
sort = dtr.feature_importances_.argsort()
plt.barh(features[sort], dtr.feature_importances_[sort], height=0.8, left=None, align='center')
plt.xlabel("Feature Importance")

params = {
    'figure.figsize': '5, 5'
}
plt.rcParams.update(params)
