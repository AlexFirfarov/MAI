import argparse
import json
import sympy
import matplotlib.pyplot as plt
from numpy import *
from sympy.parsing.sympy_parser import parse_expr


def newton_method(equation, a, b, eps):
    x = sympy.Symbol('x')
    exp = parse_expr(equation)
    func = sympy.lambdify(x, exp)

    exp_der1 = exp.diff(x)
    exp_der2 = exp_der1.diff(x)
    func_der1 = sympy.lambdify(x, exp_der1)
    func_der2 = sympy.lambdify(x, exp_der2)

    iteration = 0
    x_prev = 0
    if func(a) * func_der2(a) > 0:
        x_prev = a
    elif func(b) * func_der2(b) > 0:
        x_prev = b

    while True:
        iteration += 1
        x_cur = x_prev - func(x_prev) / func_der1(x_prev)

        if abs(x_cur - x_prev) < eps:
            print('Итераций: ', iteration)
            return x_cur
        x_prev = x_cur


def method_of_simple_iteration(fi, a, b, eps):
    x = sympy.Symbol('x')
    exp_fi = parse_expr(fi)
    exp_fi_der1 = exp_fi.diff(x)

    func_fi = sympy.lambdify(x, exp_fi)
    func_fi_der1 = sympy.lambdify(x, exp_fi_der1)

    q = max(abs(func_fi_der1(a)), abs(func_fi_der1(b)))
    assert q < 1

    iteration = 0
    x_prev = (a + b) / 2

    while True:
        iteration += 1
        x_cur = func_fi(x_prev)

        if q * abs(x_cur - x_prev) / (1 - q) <= eps:
            print('Итераций: ', iteration)
            return x_cur
        x_prev = x_cur


def graph(equation):
    func = lambda x: eval(equation)
    x0 = lambda x: x * 0

    plt.subplots()
    x = linspace(-1.0, 1.0, 1000000)
    plt.plot(x, func(x))
    plt.plot(x, x0(x))
    plt.grid()
    plt.show()


def test(eq, res):
    print('Проверка')
    x = sympy.Symbol('x')
    exp = parse_expr(eq)
    func = sympy.lambdify(x, exp)

    print('{0:.16f}'.format(func(res)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    while True:
        method = int(input('Метод простых итераций - 1\nМетод Ньютона - 2\n'))
        if method in (1, 2):
            break

    with open(args.input, 'r') as f:
        data = json.load(f)
        Eps = float(data['eps'])
        Equation = data['equation']
        Interval = data['interval']

        if method == 1:
            Fi = data['fi']

    graph(Equation)

    if method == 1:
        X = method_of_simple_iteration(Fi, *Interval, Eps)
    elif method == 2:
        X = newton_method(Equation, *Interval, Eps)

    print('x = {0}'.format(X))
    test(Equation, X)
