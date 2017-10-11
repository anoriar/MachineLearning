# coding=utf-8
import pandas
import sklearn
import sklearn.datasets
from sklearn import model_selection
from numpy import linspace
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

import sys
sys.path.append("..")

data = sklearn.datasets.load_boston()
X = data.data
y = data.target

X = sklearn.preprocessing.scale(X)

def find_p(kf, X, y):
    scores = list()
    p_range = linspace(1, 10, 200)
    for p in p_range:
        scores.append(sklearn.model_selection.cross_val_score(estimator=KNeighborsRegressor(p=p, n_neighbors=5, weights='distance'),
                                                           X=X, y=y, cv=kf, scoring='neg_mean_squared_error'))

    df = pandas.DataFrame(scores, p_range).max(axis=1).sort_values(ascending=False)
    return df.head(1).index[0]

kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
print("p до нормализации признаков: {}". format(find_p(kf, X, y)))
X = sklearn.preprocessing.scale(X)
print("p после нормализации признаков: {}". format(find_p(kf, X, y)))
