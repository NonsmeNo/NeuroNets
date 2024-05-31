import numpy as np
import math


def norm_inputs(inputs):
    """Нормализация входных значений"""
    for i in range(len(inputs)):
        for j in range(2, 7):
            inputs[i][j] /= 100
    return inputs

class Neuron:
    def __init__(self):
        """Инициализация нейрона"""
        self.weights = np.random.rand(7)

    def get_distance(self, input):
        """Расстояние от текущего входного вектора до конкретного нейрона"""
        summ = 0
        for i in range(len(input)-1):
            summ += (input[i] - self.weights[i]) ** 2

        return math.sqrt(summ)

    def correct_weights(self, n, input):
        """Корректировка весов по формуле"""
        for i in range(len(self.weights)):
            self.weights[i] += n * (input[i] - self.weights[i])


class Network:
    def __init__(self, neurons, inputs):
        """Инициализация нейронами и входными значениями"""
        self.neurons = neurons
        self.inputs = inputs

    def find_winner(self, input):
        """Поиск нейрона-победителя"""
        distances = []
        for neuron in self.neurons:
            distances.append(neuron.get_distance(input))

        min_distance = min(distances)
        min_index = distances.index(min_distance)

        return min_index

    def train(self, inputs, learning_rate, steps):
        """Обучение нейросети"""
        for i in range(steps):
            for input in inputs:
                winner_index = self.find_winner(input)
                # корректируем веса нейрона-победителя
                self.neurons[winner_index].correct_weights(learning_rate, input)


            # уменьшаем скорость обучения
            learning_rate -= 0.05

    def test(self, inputs):
        """Тестируем работу нейросети"""
        print("Тестируем работу нейросети")
        for input in inputs:
            print()
            print("Входные данные:")
            print(input)
            winner_index = self.find_winner(input)
            print("Ответ нейросети (номер группы): ", winner_index)


inputs = [
    [1, 1, 60, 79, 60, 72, 63, 1.00],
    [1, 0, 60, 61, 30, 5, 17, 0.00],
    [0, 0, 60, 61, 30, 66, 58, 0.00],
    [1, 1, 85, 78, 72, 70, 85, 1.25],
    [0, 1, 65, 78, 60, 67, 65, 1.00],
    [0, 1, 60, 78, 77, 81, 60, 1.25],
    [0, 1, 55, 79, 56, 69, 72, 0.00],
    [1, 0, 55, 56, 50, 56, 60, 0.00],
    [1, 0, 55, 60, 21, 64, 50, 0.00],
    [1, 0, 60, 56, 30, 16, 17, 0.00],
    [0, 1, 85, 89, 85, 92, 85, 1.75],
    [0, 1, 60, 88, 76, 66, 60, 1.25],
    [1, 0, 55, 64, 0, 9, 50, 0.00],
    [0, 1, 80, 83, 62, 72, 72, 1.25],
    [1, 0, 55, 10, 3, 8, 50, 0.00],
    [0, 1, 60, 67, 57, 64, 50, 0.00],
    [1, 1, 75, 98, 86, 82, 85, 1.50],
    [0, 1, 85, 85, 81, 85, 72, 1.25],
    [1, 1, 80, 56, 50, 69, 50, 0.00],
    [1, 0, 55, 60, 30, 8, 60, 0.00],
]

# нормализуем входные значения
inputs = norm_inputs(inputs)


steps = 6
learning_rate = 0.30

neurons = [Neuron() for i in range(4)]
network = Network(neurons, inputs)
network.train(inputs, learning_rate, steps)

test_inputs = [
    [1, 0, 56, 55, 50, 56, 60, 0.00],
    [1, 0, 54, 55, 50, 56, 60, 0.00],
    [1, 0, 54, 54, 50, 34, 60, 0.00],
    [0, 1, 85, 85, 81, 72, 85, 1.25],
    [0, 1, 60, 88, 76, 66, 60, 1.25],
    [0, 1, 80, 83, 62, 72, 72, 1.25],
    [1, 1, 75, 86, 98, 82, 85, 1.50],
    [1, 1, 75, 98, 86, 82, 85, 1.50],
    [1, 1, 75, 98, 86, 82, 8, 1.50]
]

network.test(norm_inputs(test_inputs))