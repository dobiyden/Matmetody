import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from sympy import lambdify, parse_expr, symbols, Eq, solve as sym_solve


res = {
    'i_str': 'x * p + y * q'
}

variant = [
    [9, 27, 120],
    [19, 28, 150]
]


def read():
    auto = input('Введите "auto", если хотите ввести данные из варианта: ').strip()
    if auto == 'auto':
        res.update({
            'u_str': '5 * (x + 9) ** (1 / 2) * (y + 5) ** (1/2) + 4',
            'p': 19,
            'q': 13,
            'i': 2574,
            'lim': '0 0'
        })
    else:
        res.update({
            'u_str': input('Введите u: ').strip(),
            'p': int(input('Введите p: ').strip()),
            'q': int(input('Введите q: ').strip()),
            'i': int(input('Введите i: ').strip()),
            'lim': input('Введите границы: ').strip()
        })
    res['lim'] = list(map(int, res['lim'].split(' '))) if res['lim'] != '' else [-1000] * 2


def solve():
    x, y = symbols('x, y')
    u_scope = {"x": x, "y": y}
    res["u_sym"] = parse_expr(res["u_str"], local_dict=u_scope, evaluate=False)

    p, q = symbols('p q')
    i_scope = {"p": p, "q": q}
    i_expr = parse_expr(res["i_str"], local_dict=i_scope, evaluate=False)
    res["i_sym"] = i_expr.subs([(p, res['p']), (q, res['q'])])

    u_func = lambdify(symbols('x y'), res["u_sym"])
    i_func = lambdify(symbols('x y'), res["i_sym"])

    m = minimize(
        lambda args: -u_func(*args),
        np.zeros((1, 2)),
        constraints={'type': 'ineq', 'fun': lambda args: -i_func(*args) + res['i']},
        bounds=[(bound, np.inf) for bound in res['lim']])

    res['u_p'] = m.x.tolist()
    res['u_v'] = -m.fun


def write():
    print(f"Точка {res['u_p']}. Значение {res['u_v']}")


def draw():
    x, y = symbols('x y')
    u_func = lambdify((x, y), res['u_sym'])

    x_s = np.outer(np.linspace(res['lim'][0], 200, 100), np.ones(100))
    y_s = np.outer(np.linspace(res['lim'][1], 200, 100), np.ones(100)).T
    u_s = np.vectorize(u_func)(x_s, y_s)

    x_c = np.linspace(res['lim'][0], 200, 100)
    y_c = np.linspace(res['lim'][1], 200, 100)
    u_c = np.vectorize(u_func)(x_c, y_c.reshape((-1, 1)))

    y_func = lambdify(x, sym_solve(Eq(res['i_sym'], res['i']), y)[0])
    x_func = lambdify(y, sym_solve(Eq(res['i_sym'], res['i']), x)[0])
    x_lim = np.linspace(
        max(x_func(200), res['lim'][0]),
        min(x_func(res['lim'][1]), 200),
        100)
    y_lim = np.vectorize(y_func)(x_lim)
    u_lim = np.vectorize(u_func)(x_lim, y_lim)

    y_func = lambdify(x, sym_solve(Eq(res['u_sym'], res['u_v']), y)[0])
    x_func = lambdify(y, sym_solve(Eq(res['u_sym'], res['u_v']), x)[0])
    x_point = np.linspace(
        max(x_func(200), res['lim'][0]),
        min(x_func(res['lim'][1]), 200),
        100)
    y_point = np.vectorize(y_func)(x_point)
    u_point = np.vectorize(u_func)(x_point, y_point)

    data = [
        go.Surface(x=x_s, y=y_s, z=u_s),
        go.Scatter3d(x=[res['u_p'][0]], y=[res['u_p'][1]], z=[res['u_v']]),
        go.Scatter3d(x=x_lim, y=y_lim, z=u_lim, mode='lines'),
        go.Scatter3d(x=x_point, y=y_point, z=u_point, mode='lines')
    ]
    go.Figure(data=data).show()

    data = [
        go.Contour(x=x_c, y=y_c, z=u_c),
        go.Scatter(x=[res['u_p'][0]], y=[res['u_p'][1]]),
        go.Scatter(x=x_lim, y=y_lim, mode='lines', fill="tozeroy"),
        go.Scatter(x=x_point, y=y_point, mode='lines')
    ]
    go.Figure(data=data).show()


if __name__ == '__main__':
    read()
    solve()
    write()
    draw()
