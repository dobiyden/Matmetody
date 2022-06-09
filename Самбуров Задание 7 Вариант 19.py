import numpy as np
import pandas as pd
from scipy.optimize import Bounds, linprog
np.set_printoptions(suppress=True)

A = [280, 390, 270]
B = [160, 150, 210, 270, 150]
C = [[9, 17, 8, 10, 4], [16, 11, 12, 13, 5], [6, 9, 16, 18, 21]]

SA = sum(A)
SB = sum(B)
print("Данные варианта")
print(f"Запасы: {A}")
print(f"Потребности: {B}")
tabl = np.array(C)
print(f"Матрица C:\n{tabl}")
print('--------------------------------------------------------------------------------------------------------------------------------')

print ("сумма A = ", SA)
print ("сумма B = ", SB)
if SA == SB:
    print('Задача закрытого типа, так как суммы запасов и потребностей равны')
else:
    print('Задача открытого типа, так как суммы запасов и потребностей неравны')
    
    
A1 = A.copy()
B1 = B.copy()
C1 = [[0, 0, 0, 0 ,0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
def search_min():
    cmin = np.Inf
    c = [-1, -1]
    for i in range(0, len(A1)):
        for j in range(0, len(B1)):
            if A1[i] > 0 and B1[j] > 0 and C[i][j] < cmin:
                cmin = C[i][j]
                c = [i, j]
    return c

count = 0
summ = 0
while True:
    c = search_min()
    if c[0] == -1:
        break
    k = min(A1[c[0]], B1[c[1]])
    C1[c[0]][c[1]] = k
    A1[c[0]] -= k
    B1[c[1]] -= k
    count += 1
    summ += C[c[0]][c[1]] * C1[c[0]][c[1]]

if count == len(A) + len(B) - 1:
    print("Опорный план является невырожденным")
else:
    print("Опорный план является вырожденным")

print("План перевозки груза")
tabl1 = np.array(C1)
print(tabl1)
print("Значение целевой функции = ", summ)
print('--------------------------------------------------------------------------------------------------------------------------------')


C2 = []
aub = []
aeq = []

for i in range(0, len(C)):
    for j in range(0, len(C[i])):
        C2.append(C[i][j])

for i in range(0, len(A)):
    d = []
    for j in range(0, len(A)*len(B)):
        if j >= i*len(B) and j < (i+1)*len(B):
            d.append(1)
        else:
            d.append(0)
    aub.append(d.copy())
        
for i in range(0, len(B)):
    d = []
    for j in range(0, len(A)*len(B)):
        if j % len(B) == i:
            d.append(1)
        else:
            d.append(0)
    aeq.append(d.copy())

res = linprog(C2, aub, A, aeq, B, options={'disp': True})
arr = res.x
ans = []
for i in range(0, len(A)):
    d = []
    for j in range(0, len(B)):
        d.append(round(arr[i*len(B)+j],3))
    ans.append(d.copy())
    
min_summ = round(res.fun, 3)
print('--------------------------------------------------------------------------------------------------------------------------------')
print("Оптимальный план перевозки груза")
tabl2 = np.array(ans)
print(tabl2)
print("Оптимальная стоимость перевозки = ", min_summ)