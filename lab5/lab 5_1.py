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

        # рассчитывается новое состояние нейронов
        # и значение активационной функции signum
        new_state = np.sign(np.dot(self.weights, state))

        while not np.array_equal(new_state, state): #сравниваем состояния
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

#считаем веса один раз
#так как обучение сети проводится за одну эпоху
network.calculate_weights(examples)

def add_loud(example, loud_percent):
    col_pixels = int(len(example) * loud_percent)
    rand_pixels = np.random.choice(len(example), col_pixels)
    loud_example = example.copy()
    loud_example[rand_pixels] *= -1
    return loud_example

#графики
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for i, ax in enumerate(axes.flat):
    ax.axis('on')
    for frame in ax.spines.values():
        frame.set_visible(True)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

fig.subplots_adjust(wspace=0.1, hspace=0.1)
fig.suptitle('Примеры с шумом 10-40% и ответы', fontsize=16, color='blue', x=0.5, y=0.95)
fig.canvas.manager.set_window_title('Собачки и буквы И')

def testing(j, colors):
    noise = 0.1
    for i in range(0, 4):
        print(j)
        example = add_loud(examples[j].copy(), noise)
        output = network.recalculate(example)

        axes[i][j*2].set_ylabel(str(int(noise*100)) + '%', rotation=90, fontsize=14, labelpad=20)
        axes[i][j*2].imshow(example.reshape((10, 10)), cmap=colors)
        axes[i][j*2+1].set_ylabel("Ответ", rotation=90, fontsize=14, labelpad=20)
        axes[i][j*2+1].imshow(output.reshape((10, 10)), cmap=colors)
        noise += 0.1


testing(0, "viridis")
testing(1, "plasma")

plt.show()
