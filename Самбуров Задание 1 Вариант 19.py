import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


res = dict()

c_variant = [
    [9, 27, 120],
    [19, 28, 150]
]

v_variant = [
    2, 1
]


def read():
    auto = input('Введите "auto", если хотите ввести данные из варианта: ').strip()
    if auto == 'auto':
        c = c_variant
        v = v_variant
    else:
        c = []
        n = int(input("Количество отраслей: "))
        for i in range(n):
            row = input(f"Введите строку {n}: ").strip()
            c.append(list(map(float, row.split(' '))))
        row = input("Введите вектор изменений: ").strip()
        v = list(map(float, row.split(' ')))
    c = np.array(c)
    v = np.array(v)
    res['C'] = c[:, :len(c)]
    res['Y'] = c[:, -1]
    res['V'] = v


def solve():
    size = len(res['C'])
    C = res['C']
    Y = res['Y']
    X = C.sum(axis=1).reshape(-1, 1) + Y
    A = C[:, :size] / X.T
    H = np.linalg.inv(np.eye(size) - A)
    V = res['V']
    Y1 = Y.reshape((-1, 1)) * (1 + V)
    X1 = H @ Y1
    D = (X1 - X) / X * 100
    XC = X1 - X1 * A.sum(axis=0).reshape(-1, 1)
    res['X'] = X
    res['A'] = A
    res['H'] = H
    res['Y1'] = Y1
    res['X1'] = X1
    res['D'] = D
    res['XC'] = XC


def write():
    print("Валовый выпуск")
    print(pd.DataFrame(res['X']))
    print("Прямые затраты")
    print(pd.DataFrame(res['A']))
    print("Полные затраты")
    print(pd.DataFrame(res['H']))
    print("Конечное потребление")
    print(pd.DataFrame(res['Y1']))
    print("Валовый выпуск после изменений")
    print(pd.DataFrame(res['X']))
    print("Изменение валового выпуска")
    print(pd.DataFrame(res['D']))
    print("Чистая продукция")
    print(pd.DataFrame(res['XC']))


def draw():
    X = res['X'].T[0]
    X1 = res['X1'].T[0]
    size = len(X)
    index = np.arange(size)
    plt.title('Change of consumption')
    plt.axis([-0.5, size - 0.5, 0, 1000])
    plt.xticks(index, range(size))
    plt.bar(index, X, color='red', alpha=0.7)
    plt.bar(index, X1 - X, color='blue', alpha=0.5, bottom=X)
    plt.show()

    data = np.abs((X1 - X) / X * 100)
    index = np.arange(size)
    plt.title('Delta gross production')
    plt.axis([-0.5, size - 0.5, 0, 500])
    plt.xticks(index, range(size))
    plt.bar(index, data, color='red', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    read()
    solve()
    write()
    draw()