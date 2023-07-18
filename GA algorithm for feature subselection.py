import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
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
data = pd.read_csv('2023-01-10 Data for ML correct1.csv', usecols=range(1, 37))
label = pd.read_csv('2023-01-10 Data for ML correct1.csv', usecols=[0])

scores_r2 = []  # Cross-Validation R2 scores
scores_mean_squared_error = []  # Cross-Validation mean squared error scores
Select_time = []  # Count of times a feature is selected by the genetic algorithm

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=5)

# Linear Regression Classifier
clf = LinearRegression(n_jobs=-1)
model_name = clf.__class__.__name__
csv_filename = f"GA_output_{model_name}.csv"

# Fitness evaluation function
def getFitness(individual):
    for train_index, test_index in kf.split(data, label):
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

        clf.fit(X_train_oh_features, y_train.values.ravel())
        y_pred = clf.predict(X_test_oh_features)
        R_2 = metrics.r2_score(y_test, y_pred)
        mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
        scores_r2.append(R_2)
        scores_mean_squared_error.append(mean_squared_error)
        
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
toolbox.register("evaluate", getFitness)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize variables
def getHof():
    numPop = 200
    numGen = 80
    pop = toolbox.population(n=numPop)
    hof = tools.HallOfFame(numPop * numGen)

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

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

    return hof

def getMetrics(hof):
    percentileList = [i / (len(hof) - 1) for i in range(len(hof))]

    validation_R2_List = []
    individualList = []

    for individual in hof:
        validation_R2 = [getFitness(individual)[0]]
        validation_R2_List.append(validation_R2[0])
        individualList.append(individual)

    validation_R2_List.reverse()
    individualList.reverse()

    return validation_R2_List, individualList, percentileList

if __name__ == '__main__':
    individual = toolbox.individual()
    individual[:] = [1 for _ in range(len(data.columns))]
    Test_R2, Test_MSE = getFitness(individual)
    print('\nTest R2 with all features: \t' + str(Test_R2))
    print('Test MSE with all features: \t' + str(Test_MSE))

    hof = getHof()
    validation_R2_List, individualList, percentileList = getMetrics(hof)

    individualList_transpose = list(map(list, zip(*individualList)))
    for i in range(len(individualList_transpose)):
        Select_time.append(individualList_transpose[i].count(1))
    print(Select_time)

    maxValAccSubsetIndicies = [index for index in range(len(validation_R2_List)) if validation_R2_List[index] == max(validation_R2_List)]
    maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
    maxValSubsets = [[list(data)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxValIndividuals]

    print('\n---Optimal Feature Subset(s)---\n')
    for index in range(len(maxValAccSubsetIndicies)):
        print('Validation R2: \t\t' + str(validation_R2_List[maxValAccSubsetIndicies[index]]))
        print('Individual: \t' + str(maxValIndividuals[index]))
        print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
        print('Feature Subset: ' + str(maxValSubsets[index]))

    tck = interpolate.splrep(percentileList, validation_R2_List, s=5.0)
    ynew = interpolate.splev(percentileList, tck)

    e = plt.figure(1)
    plt.plot(percentileList, validation_R2_List, marker='o', color='r')
    plt.plot(percentileList, ynew, color='b')
    plt.title('Validation Set Regression R2 vs. Continuum with Cubic-Spline Interpolation')
    plt.xlabel('Population Ordered By Increasing Test Set R2')
    plt.ylabel('Validation Set R2')
    e.show()

    f = plt.figure(2)
    number = [i + 1 for i in range(len(individualList))]
    plt.scatter(number, validation_R2_List)
    plt.title('Validation Set Regression R2 vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set R2')
    plt.ylabel('Validation Set R2')
    f.show()
