from math import exp, fabs
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

class Neuron:
    center = 0

    def __init__(self, center):
        # инициализируем нейрон центром
        self.center = center

    def count(self, x):
        return exp(-((x - self.center) ** 2) / ((2 * 1.5 ** 2)))

# обучающая выборка
values = [
    [-2.0, -0.48],
    [-1.5, -0.78],
    [-1.0, -0.83],
    [-0.5, -0.67],
    [0.0, -0.20],
    [0.5, 0.70],
    [1.0, 1.48],
    [1.5, 1.17],
    [2.0, 0.20]
]

x = [value[0] for value in values]
y = np.array([[values[i][1]] for i in range(9)])

# центры (пока не понятно че это такое)
centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
# радиус тоже не понятно зачем
radius = 1.5
# тогда параметр a будет равен 0.22

# создаем нейроны
neurons = [Neuron(centers[i]) for i in range(len(centers))]

# после создания нейронов нужно создать характеристическую матрицу
# количество строк этой матрицы = количеству примеров из выборки
# количество столбцов - количество нейронов
h = np.array([[neurons[i].count(x[j]) for i in range(len(centers))] for j in range(len(x))])
print("Характеристическая матрица:")
print(h)

# теперь определяем синаптические коэффициенты выходного нейрона
# для этого используем еще одну формулу
# w = (h^T * h)^(-1) * h^T * y

# коротко о том, что обозначает каждая функция
# dot - скалярное произведение массивов
# inv - обратная матрица
# transpose - транспонированная
w = np.dot(la.inv(np.dot(h.transpose(), h)), np.dot(h.transpose(), y))
print("Весовые коэффициенты:")
print(w)

# теперь считаем выходные значения на основе весовых коэффициентов
answers = [sum([neurons[i].count(value[0]) * w[i][0] for i in range(5)]) for value in values]

# считаем среднюю относительную ошибку аппроксимации
n = len(x)
s = 0
for i in range(n):
    s += fabs(1 - (y[i][0] / answers[i]))
print("Средняя относительная ошибка аппроксимации " + str(s / n * 100) + " %")


# теперь нужно все нарисовать
plt.scatter(x, y, c="slateblue", label="Исходные точки выборки")
plt.plot(x, answers, color="darkturquoise", label="Полученная аппроксимация")
plt.legend()
plt.grid(True)
plt.title("Положение исходных точек\n  и график аппроксимации")
plt.show()

