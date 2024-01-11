import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from numpy import savetxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


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
    # {'criterion': 'friedman_mse', 'learning_rate': 2, 'max_depth': 5, 'max_features': 'log2', 'min_samples_split': 2,
    #  'n_estimators': 10}
    param_grid = {
        'n_estimators': [30,50,80,100,120,200],
        'learning_rate': [0.08,0.01,0.02, 0.05, 0.08, 0.1],
        'max_depth': [3,5, 6,8,10],
        'criterion': ["friedman_mse", "squared_error"],
        "min_samples_split" : [1,2,5,8,10],
        "max_features":["sqrt", "log2"]
    }

    X_train,X_test,y_train,y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    # Create the grid search
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    main()
    print('end')