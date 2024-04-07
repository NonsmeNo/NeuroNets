from random import *
class Point:
    classification = 0

    def __init__(self, a, b):
        self.x = [uniform(a, b) for _ in range(2)]

        if self.x[0] > self.x[1]:
            self.classification = 1
        else:
            self.classification = -1
class Perceptron:
    def __init__(self):
        self.w = [uniform(-1, 1) for _ in range(2)]
        self.learning_rate = 0.3
    def calculate(self, inputs): #взвешенная сумма
        b = 1  # порог
        return self.w[0] * inputs[0] + self.w[1] * inputs[1] + b
    def weight_correction(self, expected, output): #коррекция весов
        gradient = output - expected
        if expected == 0:
            self.w[0] -= self.learning_rate * (expected + gradient) * (1 + gradient)
        else:
            self.w[1] -= self.learning_rate * (expected + gradient) * (1 + gradient)

class Adaline:
    def __init__(self):
        self.w = [uniform(-1, 1) for _ in range(2)]
        self.learning_rate = 0.3

    def calculate(self, inputs): #взвешенная сумма
        b = 1  # порог
        return self.w[0] * inputs[0] + self.w[1] * inputs[1] + b

    def weight_correction(self, y, u, x): #коррекция весов дискретным способом
        for i in range(len(self.w)):
            self.w[i] -= self.learning_rate * (u - y) * x[i]
class Network:
    def reLU(self, summ):
        return max(0, summ)


#генерация обучающих примеров
points_init = [Point(0, 0.5) for i in range(20)]
points_coords = [[points_init[i].x[0], points_init[i].x[1]] for i in range(len(points_init))]
print(points_coords)
points_result_class = [points_init[i].classification for i in range(len(points_init))]
print(points_result_class)
print()

#задание 1
print("Пункт A")
neuron = Perceptron()
network = Network()

# обучаем нейросеть, изменяем веса
epochs = 1000
for i in range(epochs):
    errors = []
    for j in range(len(points_init)): #считаем взвеш сумму для каждого примера и корректируем веса
        output = network.reLU(neuron.calculate(points_coords[j]))
        neuron.weight_correction(points_result_class[j], output)

points_random = [Point((-0.5), 0.5) for i in range(1000)]
points_random_coords = [[points_random[i].x[0], points_random[i].x[1]] for i in range(len(points_random))]
points_random_result = [points_random[i].classification for i in range(len(points_random))]

print("Классификация точек после обучения:")
correctly = 0
for i in range(len(points_random)):
    output = network.reLU(neuron.calculate(points_random_coords[i]))

    classification = (-1) if output < 0.5 else 1

    if i % 100 == 0:
        print()
        print("Точка " + str(i) + ": " +str(points_random_coords[i]) + ", ответ нейросети: класс " + str(classification))

    if points_random[i].classification == classification:
        correctly += 1
print()
print("Точность персептрона: " + str(correctly) + "/1000")

print()
print("Пункт В")
neuron = Adaline()
network = Network()

# обучаем нейросеть, изменяем веса
for i in range(epochs):
    errors = []
    for j in range(len(points_init)): #считаем взвеш сумму для каждого примера и корректируем веса
        output = network.reLU(neuron.calculate(points_coords[j]))
        neuron.weight_correction(points_result_class[j], output, points_coords[j])

points_random = [Point((-0.5), 0.5) for i in range(1000)]
points_random_coords = [[points_random[i].x[0], points_random[i].x[1]] for i in range(len(points_random))]
points_random_result = [points_random[i].classification for i in range(len(points_random))]

print("Классификация точек после обучения:")

correctly = 0
for i in range(len(points_random)):
    output = network.reLU(neuron.calculate(points_random_coords[i]))

    classification = (-1) if output < 0.5 else 1

    if i % 100 == 0:
        print()
        print("Точка " + str(i) + ": " +str(points_random_coords[i]) + ", ответ нейросети: класс " + str(classification))

    if points_random[i].classification == classification:
        correctly += 1
print()
print("Точность персептрона: " + str(correctly) + "/1000")

