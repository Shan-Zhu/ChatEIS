#coding: utf-8

import numpy as np
import pandas as pd
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, ParameterGrid
from sklearn.metrics import f1_score,accuracy_score

data_input = pd.read_csv('eis_all_data-20.csv', sep=',')

labels = data_input['Type']
features = data_input.drop('Type', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
}

XGB = XGBClassifier()

results_df = pd.DataFrame(columns=['n_estimators', 'max_depth', 'learning_rate', 'mean_test_score'])

for i, params in enumerate(ParameterGrid(param_grid)):
    print(f'Running GridSearchCV {i + 1}/{len(ParameterGrid(param_grid))}...')
    XGB.set_params(**params)
    grid_search = GridSearchCV(XGB, param_grid={},
                               cv=5, n_jobs=4, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    results_df = results_df.append({
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'mean_test_score': grid_search.best_score_
    }, ignore_index=True)

results_df.to_csv('grid_search_results.csv', index=False) 
