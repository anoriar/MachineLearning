#  coding=utf-8
import pandas
import sklearn
from sklearn import model_selection
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append("..")

df = pandas.read_csv('wine.data', header=None)
print("Классы")
y = df[0]
print(y)
print("Признаки")
X = df.loc[:, 1:]
print(X)

kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)

def find_k(kf, X, y):
    k = 0
    max_accuracy = 0.0
    for i in range(1, 100):
        accuracy = sklearn.model_selection.cross_val_score(estimator=KNeighborsClassifier(n_neighbors=i),
                                                           X=X, y=y, cv=kf, scoring='accuracy').mean()

        if max_accuracy <= accuracy:
            max_accuracy = accuracy
            k = i
    return k, max_accuracy

print("Оптимальное значение параметра k до нормализации признаков: {}". format(find_k(kf=kf, X=X, y=y)))

X = sklearn.preprocessing.scale(X)
print("Оптимальное значение параметра k после нормализации признаков: {}". format(find_k(kf=kf, X=X, y=y)))
