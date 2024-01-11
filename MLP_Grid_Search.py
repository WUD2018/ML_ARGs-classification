import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def readData(filename):

    "read data from resistome"

    #'resistome.type.rf.data.txt'
    data = pd.read_csv(filename, sep ='\\\t')
    # data = data.drop(['SampleID'],axis=1)
    grp = pd.unique(data['EnvSeason'])
    X = data[data.columns[2:]]
    label = data[data.columns[1]]

    return grp,X, label


def main():

    grp, X, label = readData('ML.table.txt')

    # Hyperparameters and their potential values
    param_grid = {
        'hidden_layer_sizes': [(25,),(50,), (100,),(150,),(50, 50), (100, 50),(100,50,25)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'alpha': [0.001, 0.01,0.05],
        'learning_rate_init': [ 0.01, 0.1,0.5],
        "solver":["lbfgs", "sgd", "adam"]
    }

    # Create an MLPClassifier
    mlp = MLPClassifier(max_iter=2000, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X_scaled, label, test_size=0.2, random_state=42)

    # Create the grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    main()
    print('end')