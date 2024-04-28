import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, n):  # заполняем матрицу весов нулями
        self.n = n
        self.weights = np.zeros((n, n))

    def calculate_weights(self, examples): # считаем веса по специальной формуле
        for i in examples:
            self.weights += np.outer(i, i.T)
        self.weights /= self.n
        np.fill_diagonal(self.weights, 0) # обнуляем значения весов по диагонали
        print(self.weights)

    def recalculate(self, state): # рассчитываем новое состояние нейронов, пока оно не перестанет изменяться
        new_state = np.sign(np.dot(self.weights, state))
        while not np.array_equal(new_state, state):
            state = new_state
            new_state = np.sign(np.dot(self.weights, state))
        return new_state



N = 100
network = Network(N)

dog = np.array([
    [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
    [1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
    [1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
    [1, -1, -1, -1, -1, 1, 1, 1, 1, -1],
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
    [-1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
    [-1, 1, -1, -1, 1, -1, -1, -1, -1, -1],
    [-1, 1, -1, -1, 1, -1, -1, -1, -1, -1],
    [-1, 1, 1, -1, 1, 1, -1, -1, -1, -1]
]).reshape(100)

letter = np.array([
    [1, 1, -1, -1, -1, -1, -1, 1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, -1, -1, -1, 1, 1, -1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, 1, -1, 1, 1, -1, -1, -1, 1, 1],
    [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, 1, -1, -1, -1, -1, -1, 1, 1]
]).reshape(100)


examples = [dog, letter]
network.calculate_weights(examples) #считаем веса один раз

def add_noise(array, noise_level=0.1):
    num_pixels_to_noise = int(len(array) * noise_level)
    noise_indices = np.random.choice(len(array), num_pixels_to_noise, replace=False)
    array[noise_indices] *= -1
    return array




fig, axes = plt.subplots(8,2, figsize=(20,80))

def tests(j):
    noise = 0.1
    for i in range(j*4, j*4+4):
        test = add_noise(examples[j].copy(), noise)
        pred = network.recalculate(test)

        axes[i][0].imshow(test.reshape((10,10)), cmap='binary')
        axes[i][1].imshow(pred.reshape((10,10)), cmap='binary')
        axes[i][0].title.set_text(("'И'" if j else "Пёс") + f" с шумом {noise:.1f}")
        axes[i][1].title.set_text("Предсказание")

        noise += 0.1

tests(0)
tests(1)

plt.show()
