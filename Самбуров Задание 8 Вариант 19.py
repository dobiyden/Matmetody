from scipy.optimize import linprog
from scipy import transpose
import numpy as np

c = np.array([1, 1, 1])
b = np.array([1, 1, 1])
bn = [[0, None], [0, None], [0, None]]

a = np.array([[3, 5, 1],
              [4, 0, 2],
              [0, 2, 4]])
print("Данные варианта")
print(a)
print('--------------------------------------------------------------------------------------------------------------------------------')
a_ub = -transpose(a)
res = linprog(c = c, A_ub = a_ub, b_ub = -b, bounds = bn, method='simplex')
print(res)
print("\n")
print('--------------------------------------------------------------------------------------------------------------------------------')
print("Решение двойственной задачи: ", res.x)
print("Оптимальное значение целевой функции = ", round(res.fun, 5))


sumW = sum(res.x)


y1 = round(res.x[0]/sumW, 3)
y2 = round(res.x[1]/sumW, 3)
y3 = round(res.x[2]/sumW, 3)

# Ответы
print("Цена игры для первого игрока = ", round(1/sumW, 3))
print("Оптимальная стратегия первого игрока: ", [y1, y2, y3])
print('--------------------------------------------------------------------------------------------------------------------------------')
res2 = linprog(c = -c, A_ub = a, b_ub = b, bounds = bn, method='simplex')

print(res2)
print("\n")
print('--------------------------------------------------------------------------------------------------------------------------------')
print("Решение двойственной задачи: ", res2.x)
print("Оптимальное значение целевой функции = ", round(-res2.fun, 5))

sumW2 = sum(res2.x)
z1 = round(res2.x[0]/sumW2, 3)
z2 = round(res2.x[1]/sumW2, 3)
z3 = round(res2.x[2]/sumW2, 3)
Cs = round((1/sumW2), 3)

print("Цена игры для второго игрока = ", round(1/sumW2,3))
print("Оптимальная стратегия второго игрока: ", [z1, z2, z3])
print('--------------------------------------------------------------------------------------------------------------------------------')
print("Оптимальная стратегия первого игрока: ", [y1, y2, y3])
print("Оптимальная стратегия второго игрока: ", [z1, z2, z3])
print("Цена игры = ", Cs)