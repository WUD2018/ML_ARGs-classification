import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)

    X_train,X_test,y_train,y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    # Create the grid search to get best hyperparameters
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    main()
    print('end')