import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import sympy as sym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

A = 36 
a1 = 0.5
a2 = 0.2
w1 = 4
w2 = 2
I = 200
print(f"Данные варианта: A = {A}, a1 = {a1}, a2 = {a2}, w1 = {w1}, w2 = {w2}, I = {I}")

xs, ys = sym.symbols('x y')
Q = A * xs ** a1 * ys ** a2 - w1 * xs - w2 * ys
print(Q)
print('--------------------------------------------------------------------------------------------------------------------------------')
print('Найдите максимальную прибыль и оптимальный план')
def con(xy):
    x, y = xy
    return f'w1*x + w2*y = {(w1*x + w2*y).round(3)} <= I = {I}'

def f(xy):
    x, y = xy
    return -(A * x ** a1 * y ** a2 - w1 * x - w2 * y)

def gr(xy):
    x, y = xy
    
    der = np.zeros_like(xy)
    der[0] = -eval(str(Q.diff(xs)))
    der[1] = -eval(str(Q.diff(ys)))
    
    return der

print(f"dQ/dx = {Q.diff(xs)}")
print(f"dQ/dy = {Q.diff(ys)}")

x0 = np.array([1.0, 1.0])
res_without_constr = minimize(f, x0, jac=gr, options={'disp': True})

print(f'Оптимальные значения: x = {res_without_constr.x[0].round(4)}, y = {res_without_constr.x[1].round(4)}')
print('Max значение функции:',-res_without_constr.fun)
print('--------------------------------------------------------------------------------------------------------------------------------')
print("В силу бюджетных ограничений на ресурсы может быть потрачено не более I (ден.ед.). \nНайдите максимальную прибыль при наличии бюджетных ограничений и оптимальное для производителя сочетание (x, y) количеств используемых ресурсов.")
x_min = 0
y_min = 0
x_max = np.inf
y_max = np.inf

linear_constraint = LinearConstraint ([w1, w2], 0, I)
bounds = Bounds([x_min, y_min], [x_max, y_max])
x0 = np.array([1.0, 1.0])

res = minimize(f, x0, jac=gr, constraints=linear_constraint, bounds=bounds, options={'disp': True})

u = res.x[0]
v = res.x[1]

print(res.x)
print(f'Оптимальные значения: x = {u.round(4)}, y = {v.round(4)}')
print('Max значение функции:',-res.fun)
print(con(res.x))
print('--------------------------------------------------------------------------------------------------------------------------------')
#Пространственная модель функции прибыли
fig, ax = plt.subplots(figsize=(15,10), subplot_kw={"projection": "3d"})

ax.view_init(15, -60)

X = np.linspace(20, 100, 200)
Y = np.linspace(10, 40, 100)
X, Y = np.meshgrid(X, Y)
Z = -f(np.array([X,Y]))

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5)
ax.set_title('Функция полезности')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Q')
plt.show()

#Карта изоквант
fig, ax = plt.subplots(figsize=(15,10))

X = np.arange(0, 400, 1)
Y = np.arange(0, 400, 1)
X, Y = np.meshgrid(X, Y)
Z = -f(np.array([X,Y]))

ax.contourf(X, Y, Z, cmap='autumn')

adm = plt.Polygon([(x_min,y_min),(x_min, I/w2-w1*x_min/w2),(I/w1-w2*y_min/w1, y_min)], facecolor='lightgreen', edgecolor='darkgreen', alpha = 1, linewidth=3)
ax.add_patch(adm)

ax.set_title('Изокванты, бюджетное множество и оптимальное решение')
ax.set_xlabel('x')
ax.set_ylabel('y')

cs_m = ax.contour(X, Y, Z, levels=[-res.fun], colors='red')
cs = ax.contour(X, Y, Z, levels=[100,150,180,200], colors='black')

plt.vlines(x=u, ymin=0, ymax=v, colors='white', ls=':', lw=2)
ax.plot(u, v, 'ko')
ax.text(res.x[0], 1, f'M*({u.round(3)}, {v.round(3)})')
ax.clabel(cs)
ax.clabel(cs_m)
plt.show()
print('--------------------------------------------------------------------------------------------------------------------------------')
F = sym.Eq(Q, -res.fun)
print('Уравнение изокванты на которой достигается максимум прибыли при наличии ограничений на издержки:')
print(F)
print('Уравнение изокосты, которая соответствует ограничениям на издержки:')
print(sym.Eq(w1*xs + w2*ys, I))





