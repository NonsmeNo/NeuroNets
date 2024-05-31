
import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, n):  # заполняем матрицу весов нулями
        self.n = n
        self.weights = np.zeros((n, n))

    def calculate_weights(self, examples): # считаем веса по специальной формуле

        for i in examples:
            self.weights += np.outer(i, i.T)

        self.weights /= self.n
        np.fill_diagonal(self.weights, 0) # обнуляем значения весов по диагонали

    def recalculate(self, state): # рассчитываем новое состояние нейронов, пока оно не перестанет изменяться

        # рассчитывается новое состояние нейронов
        # и значение активационной функции signum
        new_state = np.sign(np.dot(self.weights, state))

        while not np.array_equal(new_state, state): #сравниваем состояния
            state = new_state
            new_state = np.sign(np.dot(self.weights, state))
        return new_state


class HammingNetwork:
    def __init__(self, n, p):
        self.n = n  # кол-во входов
        self.p = p # количество эталонных образцов-цифр

        self.weights1 = [] # весовые коэффициенты 1 слоя

        # рассчитываем 2 слой по специальной формуле
        self.weights2 = np.full((p, p), -1 / (p - 1))
        np.fill_diagonal(self.weights2, 1) #если i = j то 1

        # функция активации (линейная пороговая функция)
        def function(x):
            if x >= 0:
                return x
            else:
                return 0
        self.activation = function

        # максимально допустимая разница
        # между состояниями сети на двух последовательных шагах
        self.E_max = 0.1

    # матрица весовых коэффициентов первого слоя
    # на основе матрицы эталонных образов
    def calculate_weights(self, examples):
        self.weights1 = examples

    def recalculate(self, state):

        new_state = []
        weights1_state = np.dot(self.weights1, state)

        for element in weights1_state: # применяем функцию активации ко всем элементам
            result = self.activation(element)
            new_state.append(result)

        while len(state) > 10 or np.linalg.norm(
                np.array(new_state) - np.array(state)) > self.E_max:  # Новые выходные значения
            state = new_state
            weights2_state = np.dot(self.weights2, state)

            activations = []
            for element in weights2_state:  # применяем функцию активации ко всем элементам
                result = self.activation(element)
                activations.append(result)
            new_state = activations

        if sum(np.array(new_state) > 0) > 1:
            return -1
        return np.argmax(new_state)


N = 49 # картинка 7x7
P = 10 # количество эталонных образцов-цифр

#обучающие примеры
number_0 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_1 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)
number_2 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_3 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_4 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, 1, -1, -1],
    [-1, -1, 1, -1, 1, -1, -1],
    [-1, 1, -1, -1, 1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_5 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_6 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_7 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_8 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)

number_9 = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
]).reshape(N)


examples = [number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9]
hopfield_network = HopfieldNetwork(N)
hamming_network = HammingNetwork(N, P)

#считаем веса один раз
#так как обучение сети проводится за одну эпоху
hopfield_network.calculate_weights(examples)
hamming_network.calculate_weights(examples)


#пошумим
#добавляем шум 10% для тестирования
def add_loud(example):
    loud_percent = 0.1
    col_pixels = int(len(example) * loud_percent)
    rand_pixels = np.random.choice(len(example), col_pixels)
    loud_example = example.copy()
    loud_example[rand_pixels] *= -1
    return loud_example

#графики
fig, axes = plt.subplots(3, 10, figsize=(20, 20))
for i, ax in enumerate(axes.flat):
    ax.axis('on')
    for frame in ax.spines.values():
        frame.set_visible(True)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

fig.subplots_adjust(wspace=0.1, hspace=0.1)
fig.suptitle('Сравнение нейросети Хэмминга и Хопфилда', fontsize=24, x=0.5, y=0.95)
fig.canvas.manager.set_window_title('Тесты')

for i in range(P):
    example = add_loud(examples[i].copy())
    hopfield_outputs = hopfield_network.recalculate(example)
    hamming_outputs = hamming_network.recalculate(example)

    axes[0][i].imshow(example.reshape((7, 7)), cmap='binary')
    hamming_outputs = examples[hamming_outputs] if hamming_outputs != -1 else example
    axes[1][i].imshow(hamming_outputs.reshape((7, 7)), cmap='viridis')
    axes[2][i].imshow(hopfield_outputs.reshape((7, 7)), cmap='plasma')

axes[0][0].set_ylabel('Пример', rotation=90, fontsize=14, labelpad=20)
axes[1][0].set_ylabel('Хэмминга', rotation=90, fontsize=14, labelpad=20)
axes[2][0].set_ylabel('Хопфилда', rotation=90, fontsize=14, labelpad=20)
plt.show()
