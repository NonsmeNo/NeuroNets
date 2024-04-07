from random import *

class Neuron:
    w = []

    def __init__(self):
        """Create random weights"""
        self.w = [uniform(-1, 1) for i in range(2)]
        print(self.w)

    def summing(self, x):
        return self.w[0] * x[0] + self.w[1] * x[1] + 1

    def change_weights(self, n, answer, gradient):
        """Меняем веса по правилу градиентного спуска"""
        if answer == 1:
            self.w[0] -= n * (answer + gradient) * (1 + gradient)
        else:
            self.w[1] -= n * (answer + gradient) * (1 + gradient)

class Network:
    def relu(self, summ):
        return max(0, summ)

    def mse(self, answer, real_output):
        """Функция среднеквадратичной ошибки, которую будем минимизировать"""
        error = (answer - real_output) * (answer - real_output)
        return error




sites = [[0, 0], [1, 1], [1, 0], [0, 1]]
answers = [1, 1, 0, 0]
n = 0.3

neuron = Neuron()
network = Network()

# результаты нейросети до обучения
print("Результаты работы до обучения:")
for i in range(len(sites)):
    output = network.relu(neuron.summing(sites[i]))

    if output < 0.5:
        print("Точка " + str(sites[i]) + " относится к классу 0")
    else:
        print("Точка " + str(sites[i]) + " относится к классу 1")

# обучаем нейросеть, изменяем веса
print()
print("Обучение нейросети")
for i in range(1000):
    real_outputs = []
    errors = []

    for j in range(len(sites)):
        real_output = network.relu(neuron.summing(sites[j]))
        real_outputs.append(real_output)
        error = network.mse(answers[j], real_output)
        errors.append(error)

    gradient = []
    for k in range(len(answers)):
        gradient.append(real_outputs[k] - answers[k])
        # меняем веса
        neuron.change_weights(n, answers[k], gradient[k])

   # print("Эпоха обучения: " + str(i + 1) + " | Ошибки: " + str(errors))


# результаты нейросети после обучения
print()
print("Результаты работы после обучения:")
for i in range(len(sites)):
    output = network.relu(neuron.summing(sites[i]))

    if output < 0.5:
        print("Точка " + str(sites[i]) + " относится к классу 0")
    else:
        print("Точка " + str(sites[i]) + " относится к классу 1")

