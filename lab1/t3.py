#!/usr/bin/env python3
#_*_ coding: utf-8 _*_

from random import random
from math import sqrt
class Neuron:
    def __init__(self):
        self.nu = 0.5 #коэффициент обучения
        self.w = [random(), random()] #веса
        divider = sqrt(self.w[0] ** 2 + self.w[1] ** 2) #нормирование весов
        self.w[0] /= divider
        self.w[1] /= divider

    def calculate(self, x):
        return self.w[0] * x[0] + self.w[1] * x[1] #взвешенная сумма

    def recalculate(self, x, u): #правило Гроссберга
        self.w[0] += self.nu * x[0] * u
        self.w[1] += self.nu * x[1] * u

class NeuralNetwork:
    def __init__(self):
        self.x = [ #входные обучающие векторы
            [0.97, 0.2],
            [1, 0],
            [-0.72, 0.7],
            [-0.67, 0.74],
            [-0.8, 0.6],
            [0, -1],
            [0.2, -0.97],
            [-0.3, -0.95]
        ]

        self.neurons = [Neuron() for i in range(2)] #определение начальных весов

    def __str__(self):
        s = ""
        for i, neuron in enumerate(self.neurons):
            s += f'w{str(i+1)}: {neuron.w}\n'
        return s

    def start(self, threshold_number):
        u = [0 for i in range(2)]  # вектор выходных сигналов
        number_wins = [0 for i in range(2)]
        for i in range(len(self.x)):
            for j in range(2):
                if number_wins[j] < threshold_number:
                    u[j] = self.neurons[j].calculate(self.x[i])
                else:
                    u[j] -= 100
            j = u.index(max(u))
            number_wins[j] += 1
            print('x = ' + str(self.x[i]))
            print('выиграл ' + str(j + 1) + ' нейрон')
            self.neurons[j].recalculate(self.x[i], u[j])  # победитель меняет веса
            print(f'измененные веса победителя: {str(self.neurons[j].w)} \n')


nn = NeuralNetwork()
print('\n')
print('начальные веса' + '\n' + str(nn))
nn.start(3)
print('конечные веса' + '\n' + str(nn))