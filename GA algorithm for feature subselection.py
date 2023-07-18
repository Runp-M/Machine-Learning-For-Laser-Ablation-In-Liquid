#

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import random
import warnings
from deap import base, creator, tools, algorithms
from scipy import interpolate
import matplotlib.pyplot as plt

# 1. Data Retrieval
# Modify the respective addresses and feature counts in the following code to read the desired data.
data = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with the path to your data file
label = data.pop('target_variable')  # Replace 'target_variable' with the name of your target variable column

scores_r2 = []  # Cross-Validation R2 scores
scores_mean_squared_error = []  # Cross-Validation mean squared error scores
select_time = []  # Count of times a feature is selected by the genetic algorithm

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=5)

# Linear Regression Classifier
clf = LinearRegression(n_jobs=-1)
model_name = clf.__class__.__name__
csv_filename = f"GA_output_{model_name}.csv"

# Fitness evaluation function
def get_fitness(individual):
    for train_index, test_index in kf.split(data):
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        
        # Applying one-hot encoding to the categorical features
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_train_parsed = x_train.drop(x_train.columns[cols], axis=1)
        X_train_oh_features = pd.get_dummies(X_train_parsed)
        X_test_parsed = x_test.drop(x_test.columns[cols], axis=1)
        X_test_oh_features = pd.get_dummies(X_test_parsed)

        transfer = StandardScaler()
        X_train_oh_features = transfer.fit_transform(X_train_oh_features)
        X_test_oh_features = transfer.transform(X_test_oh_features)

        clf.fit(X_train_oh_features, y_train)
        y_pred = clf.predict(X_test_oh_features)
        r2 = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        scores_r2.append(r2)
        scores_mean_squared_error.append(mse)
        
    scores_r2_mean = np.mean(scores_r2)
    scores_mean_squared_error_mean = np.mean(scores_mean_squared_error)
    scores_r2.clear()
    scores_mean_squared_error.clear()
    fitness_value_1 = scores_r2_mean
    fitness_value_2 = scores_mean_squared_error_mean
    fitness_values = (fitness_value_1, fitness_value_2)
    individual.fitness.values = fitness_values
    return fitness_values

# DEAP Global Variables
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(data.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", get_fitness)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize variables
def get_hof():
    num_pop = 200
    num_gen = 80
    pop = toolbox.population(n=num_pop)
    hof = tools.HallOfFame(num_pop * num_gen)

    def min_fitness1(individuals):
        return min(ind.fitness.values[0] for ind in individuals)

    def max_fitness1(individuals):
        return max(ind.fitness.values[0] for ind in individuals)

    def avg_fitness1(individuals):
        return np.mean([ind.fitness.values[0] for ind in individuals])

    def std_fitness1(individuals):
        return np.std([ind.fitness.values[0] for ind in individuals])

    def min_fitness2(individuals):
        return min(ind.fitness.values[1] for ind in individuals)

    def max_fitness2(individuals):
        return max(ind.fitness.values[1] for ind in individuals)

    def avg_fitness2(individuals):
        return np.mean([ind.fitness.values[1] for ind in individuals])

    def std_fitness2(individuals):
        return np.std([ind.fitness.values[1] for ind in individuals])

    stats = tools.Statistics()
    stats.register("min_fitness1", min_fitness1)
    stats.register("max_fitness1", max_fitness1)
    stats.register("avg_fitness1", avg_fitness1)
    stats.register("std_fitness1", std_fitness1)
    stats.register("min_fitness2", min_fitness2)
    stats.register("max_fitness2", max_fitness2)
    stats.register("avg_fitness2", avg_fitness2)
    stats.register("std_fitness2", std_fitness2)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_gen, stats=stats, halloffame=hof, verbose=True)

    return hof

def get_metrics(hof):
    percentile_list = [i / (len(hof) - 1) for i in range(len(hof))]

    validation_r2_list = []
    individual_list = []

    for individual in hof:
        validation_r2 = [get_fitness(individual)[0]]
        validation_r2_list.append(validation_r2[0])
        individual_list.append(individual)

    validation_r2_list.reverse()
    individual_list.reverse()

    return validation_r2_list, individual_list, percentile_list

if __name__ == '__main__':
    individual = toolbox.individual()
    individual[:] = [1 for _ in range(len(data.columns))]
    test_r2, test_mse = get_fitness(individual)
    print('\nTest R2 with all features: \t' + str(test_r2))
    print('Test MSE with all features: \t' + str(test_mse))

    hof = get_hof()
    validation_r2_list, individual_list, percentile_list = get_metrics(hof)

    individual_list_transpose = list(map(list, zip(*individual_list)))
    for i in range(len(individual_list_transpose)):
        select_time.append(individual_list_transpose[i].count(1))
    print(select_time)

    max_val_acc_subset_indices = [index for index in range(len(validation_r2_list)) if validation_r2_list[index] == max(validation_r2_list)]
    max_val_individuals = [individual_list[index] for index in max_val_acc_subset_indices]
    max_val_subsets = [[data.columns[index] for index in range(len(individual)) if individual[index] == 1] for individual in max_val_individuals]

    print('\n---Optimal Feature Subset(s)---\n')
    for index in range(len(max_val_acc_subset_indices)):
        print('Validation R2: \t\t' + str(validation_r2_list[max_val_acc_subset_indices[index]]))
        print('Individual: \t' + str(max_val_individuals[index]))
        print('Number of Features in Subset: \t' + str(len(max_val_subsets[index])))
        print('Feature Subset: ' + str(max_val_subsets[index]))

    tck = interpolate.splrep(percentile_list, validation_r2_list, s=5.0)
    ynew = interpolate.splev(percentile_list, tck)

    e = plt.figure(1)
    plt.plot(percentile_list, validation_r2_list, marker='o', color='r')
    plt.plot(percentile_list, ynew, color='b')
    plt.title('Validation Set Regression R2 vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set R2')
    plt.ylabel('Validation Set R2')
    plt.show()

    f = plt.figure(2)
    number = [i + 1 for i in range(len(individual_list))]
    plt.scatter(number, validation_r2_list)
    plt.title('Validation Set Regression R2 vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set R2')
    plt.ylabel('Validation Set R2')
    plt.show()
