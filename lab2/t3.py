import random
import math

class Neuron:
    w = []

    def __init__(self):
        self.w = [random.uniform(-1, 1) for i in range(9)]

    def summing(self, x):
        summ = 1

        for i in range(9):
            summ += x[i] * self.w[i]

        return summ

    def change_weights(self, n, y, u, x_in):
        for i in range(len(self.w)):
            self.w[i] -= n * (u - y) * x_in[i]

class Network:

    def relu(self, x):
        return x

    def mse(self, answer, output):
        summ = 0

        for i in range(len(answer)):
            summ += answer[i] - output[i]

        return summ / len(answer)

def testing(x):
    for i in range(len(x)):
        outputs = []

        for j in range(len(x)):
            output = network.relu(neurons[j].summing(x[i]))
            answer = 0 if output < 0.5 else 1
            outputs.append(answer)

        print(f"Для {i + 1}-ой буквы ответ: {outputs}")


network = Network()
neurons = [Neuron() for i in range(4)]

n = 0.3
x = [[1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 1, 1]]
answers = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]

# обучение нейросети
for i in range(1000): # эпохи
    errors = []
    for j in range(len(x)): # обучающие примеры
        outputs = []

        for k in range(len(neurons)): # нейроны
            output = network.relu(neurons[k].summing(x[j]))
            outputs.append(output)

        error = network.mse(answers[j], outputs)
        errors.append(error)

        # меняем веса у нейронов
        for k in range(len(neurons)):
            neurons[k].change_weights(n, answers[j][k], outputs[k], x[j])

    if (i+1) % 50 == 0:
        print(f"Эпоха обучения: {i+1}/1000, ошибка: {errors}")


# проверка после обучения на обычных буквах
print()
print("Проверка на обыкновенных буквах:")
testing(x)

# проверка после обучения на буквах с шумами
print()
print("Проверка на буквах с шумами:")
x_loud = [[1, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0],
         [0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 1, 0]]
testing(x_loud)
