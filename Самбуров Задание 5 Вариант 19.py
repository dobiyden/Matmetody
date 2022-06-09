import numpy as np
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

a = 4
b = 8
c = 0.75
print(f"Данные варианта: а = {a}, b = {b}, c = {c}")
print('--------------------------------------------------------------------------------------------------------------------------------')
def constraint(constraint_list, cx, cy, lb, ub):
    constraint_list.append(LinearConstraint([cx, cy], lb, ub))
    
constraint_list = []
constraint(constraint_list, 0.5, 1, -5, np.inf)
constraint(constraint_list, 1.5, 1, -9, np.inf)
constraint(constraint_list, -1, 1, -b, np.inf)
constraint(constraint_list, -c, 1, -np.inf, 8*c+3)

def function_main(ar):
    return ar[0] + a*ar[1]
    

x0 = np.array([1., 1.])
dec = minimize(function_main, x0, constraints=constraint_list)

print(dec)
ans = dec['x']
ans[0] = round(-ans[0], 3)
ans[1] = round(-ans[1], 3)
function_max = function_main(ans)
print('--------------------------------------------------------------------------------------------------------------------------------')
print(f"Оптимальные значения: x = {ans[0]}, y = {ans[1]}")
print("Максимальное значение функции: ", function_max)

plt.axis([-20, 20, -20, 20])

ms = 1000
x = np.array(range(-20*ms, 20*ms+1))/ms
# 2y <= 10 - x
y1 = 5 - x/2

# 2y <= 18 - 3x 
y2 = 9 - x*3/2

# y <= x + b
y3 = b + x

# y >= cx - 8c - 3 
y4 = x*c - (8*c+3)

fm = (function_max - x)/2
f3 = (3 - x)/2


#Отрисовка линий
plt.plot(x, y1, 'brown', label = r'$2y\leq 10-x$')
plt.plot(x, y2, 'green', label = r'$2y\leq 18-3x$')
plt.plot(x, y3, 'red', label = fr'$y\leq x + {b} $')
plt.plot(x, y4, 'blue', label = fr'$y\geq {c}(x-8)-3$')   

# M
plt.plot(ans[0], ans[1], 'ko')
plt.text(ans[0], ans[1] + 2, f'M ({ans[0]}, {ans[1]})')

plt.fill_between(x, y4, np.min([y1, y2, y3], axis = 0), where = (y4 <= y2), color = 'b', alpha = 0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()