from random import *

class Neuron:
    def __init__(self):
        self.w = [uniform(-1, 1) for _ in range(2)]
        self.learning_rate = 0.5

    def calculate(self, inputs): #взвешенная сумма
        b = 1  # порог
        return self.w[0] * inputs[0] + self.w[1] * inputs[1] + b

    def weight_correction(self, expected, output): #коррекция весов
        gradient = output - expected
        if expected == 1:
            self.w[0] -= self.learning_rate * (expected + gradient) * (1 + gradient)
        else:
            self.w[1] -= self.learning_rate * (expected + gradient) * (1 + gradient)


class Network:
    def reLU(self, x):
        return max(0, x)

inputs = [[0, 0], [1, 1], [1, 0], [0, 1]]
expected_answ = [0, 0, 1, 1]

neuron = Neuron()
network = Network()

print("Результаты работы до обучения:")
for i in range(len(inputs)):
    output = network.reLU(neuron.calculate(inputs[i]))
    if output <= 0.5:
        print("Точка " + str(inputs[i]) + " относится к классу 0")
    else:
        print("Точка " + str(inputs[i]) + " относится к классу 1")

# обучение
epochs = 1000
for i in range(epochs):
    for j in range(len(inputs)): #считаем взвеш сумму для каждого примера и корректируем веса
        output = network.reLU(neuron.calculate(inputs[j]))
        neuron.weight_correction(expected_answ[j], output)

print("\nРезультаты работы после обучения:")
for i in range(len(inputs)):
    output = network.reLU(neuron.calculate(inputs[i]))
    if output <= 0.5:
        print("Точка " + str(inputs[i]) + " относится к классу 0")
    else:
        print("Точка " + str(inputs[i]) + " относится к классу 1")
