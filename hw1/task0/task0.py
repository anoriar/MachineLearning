# coding=utf-8
import numpy as np
a = np.random.normal(1, 100, (1000, 50))
mean = np.mean(a, 0)
std = np.std(a, 0)
print("Сгенерированная матрица {}". format(a))

for col in range(a.shape[1]):
    a[:, col] = (a[:, col] - mean[col]) / std[col]
print("Нормализованная матрица = {}". format(a))

print("Номера строк, в которых сумма элементов больше 10")
sum = np.sum(a, 1)
for i, elem in enumerate(sum):
    if elem > 10:
        print(i)

array = np.array([1, 1, 1])
matrix1 = np.diag(array)
matrix2 = np.diag(array)
print("Обьединение матриц {}" . format(np.concatenate((matrix1, matrix2))))


