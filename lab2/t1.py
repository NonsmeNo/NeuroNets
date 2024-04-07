from random import *
class Neuron:
    nu = 0.5  # коэффициент обучения
    def __init__(self):
        self.w = [uniform(-1, 1) for i in range(2)]
    def calculate(self, x):
        b = 1
        return self.w[0] * x[0] + self.w[1] * x[1] + b

    def change_weights(self, answer, gradient):
        if answer == 1:
            self.w[0] -= self.nu * (answer + gradient) * (1 + gradient)
        else:
            self.w[1] -= self.nu * (answer + gradient) * (1 + gradient)


inputs = [[0, 0], [1, 1], [1, 0], [0, 1]]
answers = [1, 1, 0, 0]

class Network:
    def reLU(self, summ):
        return max(0, summ)

    def mse(self, answer, real_output):
        """Функция среднеквадратичной ошибки, которую будем минимизировать"""
        error = (answer - real_output) * (answer - real_output)
        return error

neuron = Neuron()
network = Network()

# результаты нейросети до обучения
print("Результаты работы до обучения:")
for i in range(len(inputs)):
    output = network.reLU(neuron.calculate(inputs[i]))

    if output < 0.5:
        print("Точка " + str(inputs[i]) + " относится к классу 0")
    else:
        print("Точка " + str(inputs[i]) + " относится к классу 1")

# обучаем нейросеть, изменяем веса
print()
print("Обучение нейросети")
for i in range(1000):
    real_outputs = []
    errors = []

    for j in range(len(inputs)):
        real_output = network.reLU(neuron.calculate(inputs[j]))
        real_outputs.append(real_output)
        error = network.mse(answers[j], real_output)
        errors.append(error)

    gradient = []
    for k in range(len(answers)):
        gradient.append(real_outputs[k] - answers[k])
        # меняем веса
        neuron.change_weights(answers[k], gradient[k])

    print("Эпоха обучения: " + str(i + 1) + " | Ошибки: " + str(errors))


# результаты нейросети после обучения
print()
print("Результаты работы после обучения:")
for i in range(len(inputs)):
    output = network.reLU(neuron.calculate(inputs[i]))

    if output < 0.5:
        print("Точка " + str(inputs[i]) + " относится к классу 0")
    else:
        print("Точка " + str(inputs[i]) + " относится к классу 1")
