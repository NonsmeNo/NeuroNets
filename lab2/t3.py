import random
import math

from random import *

class Neuron:
    def __init__(self):
        self.w = [uniform(-1, 1) for _ in range(9)]
        self.learning_rate = 0.5

    def calculate(self, inputs): #взвешенная сумма
        b = 1  # порог
        for i in range(9):
            b += inputs[i] * self.w[i]
        return b

    def weight_correction(self, expected, output, inputs):
        for i in range(len(self.w)):
            self.w[i] -= self.learning_rate * (output - expected) * inputs[i]

def testing(x):
    print (x)
    for i in range(len(x)):
        outputs = []

        for j in range(len(x)):
            output = network.reLU(neurons[j].calculate(x[i]))
            answer = 0 if output < 0.5 else 1
            outputs.append(answer)

        print(f"Для {i + 1}-ой буквы ответ: {outputs}")
class Network:
    def reLU(self, x):
        return max(0, x)


neurons = [Neuron() for i in range(4)]
network = Network()

inputs = [[1, 0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 0, 1, 0],
          [0, 1, 0, 0, 1, 0, 0, 1, 0],
          [1, 0, 0, 1, 0, 0, 1, 1, 1]]

expected_answ = [[0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [1, 0, 0, 0]]
# обучение
epochs = 5000
for i in range(epochs):
    for j in range(len(inputs)): # обучающие примеры
        for k in range(len(neurons)): # нейроны
            output = network.reLU(neurons[k].calculate(inputs[j]))
            neurons[k].weight_correction(expected_answ[j][k], output, inputs[j])



# проверка после обучения на обычных буквах
print()
print("Проверка на обыкновенных буквах:")
testing(inputs)

# проверка после обучения на буквах с шумами
print()
print("Проверка на буквах с шумами:")
x_loud = [[1, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0],
         [0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 1, 0]]
testing(x_loud)
