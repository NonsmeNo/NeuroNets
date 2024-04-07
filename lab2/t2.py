from random import *

class Site:
    x = []
    clas = 0

    def __init__(self, a, b, fl):
        self.x = [uniform(a, b) for i in range(2)]

        if fl:
            self.clas = 1 if self.x[0] > self.x[1] else (-1)
        else:
            pass


class Neuron_Adaline:
    w = []

    def __init__(self):
        """Create random weights"""
        self.w = [uniform(-1, 1) for i in range(2)]

    def summing(self, x):
        summ = 1  # порог

        for i in range(len(self.w)):
            summ += self.w[i] * x[i]

        return summ


    def change_weights(self, n, y, u, x):
        """Изменяем веса дискретным способом, по формуле"""
        # тут уже не градиентный спуск
        for i in range(len(self.w)):
            self.w[i] -= n * (u - y) * x[i]

class Neuron:
    w = []

    def __init__(self):
        """Create random weights"""
        self.w = [uniform(-1, 1) for i in range(2)]

    def summing(self, x):
        summ = 1   # порог

        for i in range(len(self.w)):
            summ += self.w[i] * x[i]

        return summ


    def change_weights(self, n, answer, gradient):
        """Меняем веса по правилу градиентного спуска"""
        if answer == 0:
            self.w[0] -= n * (answer + gradient) * (1 + gradient)
        else:
            self.w[1] -= n * (answer + gradient) * (1 + gradient)

class Network:
    def relu(self, summ):
        return max(0, summ)

    def signum(self, x):
        """Функция активации сигнум"""
        if x > 0:
            return 1
        else:
            return -1

    def mse(self, answer, real_output):
        """Функция среднеквадратичной ошибки, которую будем минимизировать"""
        error = (answer - real_output) * (answer - real_output)
        return error


sites_train = [Site(0, 0.5, True) for i in range(20)]
sites_train_cords = [[sites_train[i].x[0], sites_train[i].x[1]] for i in range(len(sites_train))]
sites_train_answer = [sites_train[i].clas for i in range(len(sites_train))]
n = 0.3

print()
print("<<<<<<<<<<<<<<<<< Task A >>>>>>>>>>>>>>>>>>>")
neuron = Neuron()
network = Network()

# обучаем нейросеть, изменяем веса
print()
print("Обучение нейросети")

for i in range(1000):
    real_outputs = []
    errors = []

    for j in range(len(sites_train)):
        real_output = network.relu(neuron.summing(sites_train_cords[j]))
        real_outputs.append(real_output)
        error = network.mse(sites_train_answer[j], real_output)
        errors.append(error)

    gradient = []
    for k in range(len(sites_train_answer)):
        gradient.append(real_outputs[k] - sites_train_answer[k])
        # меняем веса
        neuron.change_weights(n, sites_train_answer[k], gradient[k])

    if (i+1) % 100 == 0:
        print("Эпоха обучения: " + str(i + 1) + " | Ошибки: " + str(errors))

print()
sites_random = [Site((-0.5), 0.5, True) for i in range(1000)]
sites_random_cords = [[sites_random[i].x[0], sites_random[i].x[1]] for i in range(len(sites_random))]
sites_random_answer = [sites_random[i].clas for i in range(len(sites_random))]

print("Классификация точек после обучения")

rights = 0

for i in range(len(sites_random)):
    output = network.relu(neuron.summing(sites_random_cords[i]))

    clas = (-1) if output < 0.5 else 1

    if i % 200 == 0:
        print()
        print("Точка " + str(sites_random_cords[i]) + ", класс: " + str(sites_random_answer[i]))
        print("Ответ нейросети: " + str(clas))

    if sites_random[i].clas == clas:
        rights += 1

print("Точность обычного нейрона: " + str(rights) + "/1000")

print()
print()
print("<<<<<<<<<<<<<<<<< Task B >>>>>>>>>>>>>>>>>>>")
neuron = Neuron_Adaline()
network = Network()

# обучаем наш нейрон и считаем ошибки
for i in range(1000):
    real_outputs = []
    errors = []

    for j in range(len(sites_train)):
        real_output = network.signum(neuron.summing(sites_train_cords[j]))
        real_outputs.append(real_output)
        error = network.mse(sites_train_answer[j], real_output)
        errors.append(error)
        neuron.change_weights(n, sites_train_answer[j], real_output, sites_train_cords[j])


    if (i + 1) % 100 == 0:
        print("Эпоха обучения: " + str(i+1) + " | Ошибки: " + str(errors))

print("Классификация точек после обучения")

rights = 0

for i in range(len(sites_random)):
    output = network.signum(neuron.summing(sites_random_cords[i]))

    clas = (-1) if output < 0.5 else 1

    if i % 200 == 0:
        print()
        print("Точка " + str(sites_random_cords[i]) + ", класс: " + str(sites_random_answer[i]))
        print("Ответ нейросети: " + str(clas))

    if sites_random[i].clas == clas:
        rights += 1

print("Точность нейрона Адалайна: " + str(rights) + "/1000")
