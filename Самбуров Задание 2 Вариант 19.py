import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from sympy import lambdify, parse_expr, symbols


res = {
    'p_str': 'p1 * q1 + p2 * q2 - c'
}

variant = [
    [9, 27, 120],
    [19, 28, 150]
]


def read():
    auto = input('Введите "auto", если хотите ввести данные из варианта: ').strip()
    if auto == 'auto':
        res.update({
            'q1_str': '48 - p1',
            'q2_str': '116 -4 * p2',
            'c_str': '3 * q1 ** 2 + 4 * q1 * q2 + 2 * q2 ** 2 + 7',
        })
    else:
        res.update({
            'q1_str': input('Введите q1: ').strip(),
            'q2_str': input('Введите q2: ').strip(),
            'c_str': input('Введите c: ').strip(),
        })


def solve():
    p1, p2 = symbols('p1 p2')
    q_scope = {"p1": p1, "p2": p2}
    res["q1_sym"] = parse_expr(res['q1_str'], local_dict=q_scope, evaluate=False)
    res["q2_sym"] = parse_expr(res['q2_str'], local_dict=q_scope, evaluate=False)

    q1, q2 = symbols('q1 q2')
    c_scope = {"q1": q1, "q2": q2}
    c_expr = parse_expr(res['c_str'], local_dict=c_scope, evaluate=False)
    res["c_sym"] = c_expr.subs([(q1, res["q1_sym"]), (q2, res["q2_sym"])])

    c = symbols('c')
    p_scope = {**q_scope, **c_scope, "c": c}
    p_expr = parse_expr(res['p_str'], local_dict=p_scope, evaluate=False)
    res["p_sym"] = p_expr.subs([(q1, res["q1_sym"]), (q2, res["q2_sym"]), (c, res["c_sym"])])

    p_lam = lambdify(symbols('p1 p2'), res["p_sym"])
    m = minimize(lambda args: -p_lam(*args), np.zeros((1, 2)))

    res['p_p'] = m.x.tolist()
    res['p_v'] = -m.fun
    res['q_p'] = [
        res['q1_sym'].subs(symbols('p1'), res['p_p'][0]),
        res['q2_sym'].subs(symbols('p2'), res['p_p'][1])
    ]


def write():
    print(f"Точка {res['q_p']}. Значение {res['p_v']}")


def draw():
    p_func = lambdify(symbols('p1 p2'), res['p_sym'])

    p1_s = np.outer(np.linspace(0, 100, 100), np.ones(100))
    p2_s = p1_s.T
    p_s = np.vectorize(p_func)(p1_s, p2_s)

    p1_c = np.linspace(0, 100, 100)
    p2_c = p1_c
    p_c = np.vectorize(p_func)(p1_c, p2_c.reshape((-1, 1)))

    go.Figure(data=[
        go.Surface(x=p1_s, y=p2_s, z=p_s),
        go.Scatter3d(x=[res['p_p'][0]], y=[res['p_p'][1]], z=[res['p_v']]),
    ]).show()

    go.Figure(data=[
        go.Contour(x=p1_c, y=p2_c, z=p_c),
        go.Scatter(x=[res['p_p'][0]], y=[res['p_p'][1]])
    ]).show()


if __name__ == '__main__':
    read()
    solve()
    write()
    draw()