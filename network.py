import random
import json
import numpy as np


class Network():

    def __init__(self, sizes):
        """На вход подается список с двумя переменными, первая это количество признаков (фичей),
         второе число должно быть 1, так на выходе мы имеем сигмойдую. 
         Пример использования net = network.Network([4,1])
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.lerning_rate = 0.1
        self.lmbda = 0.1
        self.weight_initialization()


    def weight_initialization(self):
        """Функция np.random.randn(), выдает числа из стандартного распределения с мат. ожиданием 0 и стандартным отклонение 1.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def load_data(self, dataset):
        """Делим наш датасет на 3 части. 
        """
        len_data = len(dataset)
        np.random.shuffle(dataset)
        self.training_data = dataset[:int(len_data * 0.6)]
        self.validation_data = dataset[int(len_data * 0.6):int(len_data * 0.8)]
        self.test_data = dataset[int(len_data * 0.8):]


    def learning(self, epochs, mini_batch_size, eta, lmbda):
        ''' Функция для обучения сети'''
        for j in range(epochs):
            random.shuffle(self.training_data)
            mini_batches = [
                self.training_data[k:k+mini_batch_size]
                for k in range(0, len(self.training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.gradient_step(
                    mini_batch, eta, lmbda, len(self.training_data))

            print("Epoch %s training complete," % (j+1), " accuracy on test data - ", self.evaluate(self.test_data), "%")
        print("Accuracy on training data - ", self.evaluate(self.training_data), "%")
        print("Accuracy on validation data - ", self.evaluate(self.validation_data), "%")    

    def gradient_step(self, mini_batch, eta, lmbda, n):
        """
        Обновляем васа и смещение согласно данным посчитанным на одной мини корзине.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        """
        Выражение (1-eta*(lmbda/n)) это регуляризация, которая уменьшает значение веса.
        Функция производной от квадратичной ошибки приводит к излишнему обучению сети. Веса становятся
        большими и мы получает более высокое значение ошибки на валидационных данных.
        Это выражение близко к 0,9 - что постоянно немоного уменьшает значение веса и защищает
        от переобученности сети.
        """
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Использую обратное распространение ошибки считаем градиент для последующего изменения весов и
        смещения.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # Обратное распространени
        delta = self.cost_func(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)        

    def feedforward(self, a):
        """Считаем результат для одного наблюдения
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def cost_func(self, a, y):
        """Возьмем производную от квадратичной ошибки.
        """
        return (a-y)    

    def sigmoid(self, z):
        """Считаем сигмоиду по каждому нейрону. Сигмоида получает на вход значения от минус бесконечности,
        до плюс бесконечности. На выходе сигмоида выдает значения от 0 до 1. Основное изменение функции идет
        на участке х от -6 до 6, еще больше от -4 до 4. При "х" равном 0, сигмоида равна 0,5. В этой же точке,
        мы имеем самое больше значение производной 0.25.
        """
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Считаем производную от сигмоиды. Производная нам показывает скорость изменения функции. 
        Максимальное значение 0.25 мы получим при входящем нуле.
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))    

    def evaluate(self, data):
        """Возвращает процент правильных ответов."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        accuracy = sum(int(x == y) for (x, y) in test_results)
        return float(accuracy/len(data)*100)


    def save(self, filename):
        """Сохраняем результаты обучения."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Загружаем сохраненную нейросеть
def load(filename):
    """Возвращает экземпляр класса.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net