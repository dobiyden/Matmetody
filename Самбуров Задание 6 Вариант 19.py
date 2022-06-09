import numpy as np
import pandas as pd
from scipy.optimize import Bounds, linprog
import matplotlib.pyplot as plt
import math

a = 3
b = 2
c = 25
print(f"Данные варианта: а = {a}, b = {b}, c = {c}")
print('--------------------------------------------------------------------------------------------------------------------------------')

limit_I = 3000
limit_II = 3320
cost_A = 6*b+12
cost_B = 5*b+22
cost_C = c

resource_A = [1, 3, a]
resource_B = [6, 5, 2]
costs = np.array([cost_A, cost_B, cost_C])
resources = np.array([resource_A, resource_B])
limits = np.array([limit_I, limit_II])

def f(x1, x2, x3):
    return x1*cost_A + x2*cost_B + x3*cost_C

def is_in_lim(x1, x2, x3):
    if x1*resource_A[0] + x2*resource_A[1] + x3*resource_A[2] <= limit_I:
        if x1*resource_B[0] + x2*resource_B[1] + x3*resource_B[2] <= limit_II:
            return True
    return False

def correction(arr):
    if arr[0] == int(arr[0]) and arr[1] == int(arr[1]) and arr[2] == int(arr[2]):
        return arr
    fmax = -np.inf
    x1 = math.floor(arr[0])
    x2 = math.floor(arr[1])
    x3 = math.floor(arr[2])
    arr_cor = [0, 0, 0]
    for i1 in range(-1, 1):
        for i2 in range (-1, 1):
            for i3 in range (-1, 1):
                if is_in_lim(x1+i1, x2+i2, x3+i3):
                    ff = f(x1+i1, x2+i2, x3+i3)
                    if fmax <= ff:
                        fmax = ff
                        arr_cor = [x1, x2, x3]          
    return arr_cor

ret = linprog(-costs, resources, limits, method='simplex', options={'disp': True})
arr = correction(ret.x)
print('--------------------------------------------------------------------------------------------------------------------------------')
print(f"Оптимальные значения: x1 = {arr[0]} x2 = {arr[1]} x3 = {arr[2]}")
print("Максимальное значение функции = ", f(arr[0], arr[1], arr[2]))
print(f"Расход ресурса I = {arr[0]*resource_A[0] + arr[1]*resource_A[1] + arr[2]*resource_A[2]} из {limit_I}")
print(f"Расход ресурса II = {arr[0]*resource_B[0] + arr[1]*resource_B[1] + arr[2]*resource_B[2]} из {limit_II}")