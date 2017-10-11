# coding=utf-8
import pandas
import re

import sys
sys.path.append("..")


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data['Pclass'] = data['Pclass'].astype(object)

sex_counts = data['Sex'].value_counts()
print('Количество мужчин: {}, женщин: {}'.format(sex_counts['male'], sex_counts['female']))

surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print('Выживших: {:0.2f}%'.format(surv_percent))

ages = data['Age'].dropna()
print("Среднее возраста пассажиров: {:0.2f}. Медиана: {:0.2f}".format(ages.mean(), ages.median()))

def clean_name(name):
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    name = name.split(' ')[0].replace('"', '')

    return name

names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
print("Самое популярное имя: {}" . format(name_counts.head(1).index.values[0]))
