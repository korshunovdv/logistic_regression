
import numpy as np
from sklearn import datasets

# Иммортируем нашу нейронную сеть
import network

def preparing_data(features, targets):
    """ Функция подгототавливает данные для передачи датасета в нейронную сеть.
     Основной принцип преобразований это попарная передача фичей и таргета.
    """
    features = [np.reshape(x, (4, 1)) for x in features]
    data = zip(features, targets)
    return list(data)

# Загружаем датасет ирис
iris = datasets.load_iris()
# Согласно ТЗ мы должны взять два вида из трех. Собственное, если мы возьмем третий вид,
# нейросеть нужно совсем немного изменить, что бы от регрессии перейти к классификации.

features = iris.data[:100]       
targets = iris.target[:100]
dataset = preparing_data(features, targets)
np.random.shuffle(dataset)
# Кроме того что мы поделим датасет внутри самой сети на три части (обучение, валидация, тест),
# мы еще осавим себе 20% для дополнительного теста
dataset1 = dataset[:80]
dataset2 =dataset[80:]

# Четыре входных папаметра и одна сигмойда на выходе
net = network.Network([4,1])
# Подгружаем наш датасет
net.load_data(dataset1)
# Обучаем сетку
net.learning(epochs=10,mini_batch_size=5,eta=0.1,lmbda=0.1)
# Сохраним нашу нейросеть для что бы каждый раз не обучать ее заново, а использовать обученные веса.
net.save('test_net')

# Делаем дополнительный тест на тех 20%, который мы приберегли.
test_net = network.load('test_net')
print(f'Точтность предсказаний на тестовых данных {test_net.evaluate(dataset2)} %')

