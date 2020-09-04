import argparse
import json
import sympy
import matplotlib.pyplot as plt
from numpy import *
from sympy.parsing.sympy_parser import parse_expr


def method_of_simple_iteration(fi1, fi2, a1, b1, a2, b2, eps):
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')

    exp_fi1 = parse_expr(fi1)
    exp_fi2 = parse_expr(fi2)

    func_fi1 = sympy.lambdify([x1, x2], exp_fi1)
    func_fi2 = sympy.lambdify([x1, x2], exp_fi2)

    exp_fi1_der1_x1 = exp_fi1.diff(x1)
    exp_fi1_der1_x2 = exp_fi1.diff(x2)
    exp_fi2_der1_x1 = exp_fi2.diff(x1)
    exp_fi2_der1_x2 = exp_fi2.diff(x2)

    func_fi1_der1_x1 = sympy.lambdify([x1, x2], exp_fi1_der1_x1)
    func_fi1_der1_x2 = sympy.lambdify([x1, x2], exp_fi1_der1_x2)
    func_fi2_der1_x1 = sympy.lambdify([x1, x2], exp_fi2_der1_x1)
    func_fi2_der1_x2 = sympy.lambdify([x1, x2], exp_fi2_der1_x2)

    x_prev = [(a1 + b1) / 2, (a2 + b2) / 2]

    q = None
    for x_1 in [a1, b1]:
        for x_2 in [a2, b2]:
            q_cur = max([abs(func_fi1_der1_x1(x_1, x_2)) + abs(func_fi1_der1_x2(x_1, x_2)),
                        abs(func_fi2_der1_x1(x_1, x_2)) + abs(func_fi2_der1_x2(x_1, x_2))])
            if q is None or q_cur > q:
                q = q_cur
    assert q < 1

    iteration = 0
    while True:
        iteration += 1
        x_cur = [func_fi1(*x_prev), func_fi2(*x_prev)]

        if max([abs(x_cur[0] - x_prev[0]), abs(x_cur[1] - x_prev[1])]) * q / (1 - q) <= eps:
            print('Итераций: ', iteration)
            return x_cur
        x_prev = x_cur


def newton_method(eq1, eq2, a1, b1, a2, b2, eps):
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')

    exp1 = parse_expr(eq1)
    exp2 = parse_expr(eq2)

    func1 = sympy.lambdify([x1, x2], exp1)
    func2 = sympy.lambdify([x1, x2], exp2)

    exp1_der1_x1 = exp1.diff(x1)
    exp1_der1_x2 = exp1.diff(x2)
    exp2_der1_x1 = exp2.diff(x1)
    exp2_der1_x2 = exp2.diff(x2)

    func1_der1_x1 = sympy.lambdify([x1, x2], exp1_der1_x1)
    func1_der1_x2 = sympy.lambdify([x1, x2], exp1_der1_x2)
    func2_der1_x1 = sympy.lambdify([x1, x2], exp2_der1_x1)
    func2_der1_x2 = sympy.lambdify([x1, x2], exp2_der1_x2)

    detA1 = lambda x_1, x_2: func1(x_1, x_2) * func2_der1_x2(x_1, x_2) - func2(x_1, x_2) * func1_der1_x2(x_1, x_2)
    detA2 = lambda x_1, x_2: func2(x_1, x_2) * func1_der1_x1(x_1, x_2) - func1(x_1, x_2) * func2_der1_x1(x_1, x_2)
    detJ = lambda x_1, x_2: func1_der1_x1(x_1, x_2) * func2_der1_x2(x_1, x_2) - \
                            func2_der1_x1(x_1, x_2) * func1_der1_x2(x_1, x_2)

    x_prev = [(a1 + b1) / 2, (a2 + b2) / 2]
    iteration = 0

    while True:
        iteration += 1
        x_cur = [x_prev[0] - detA1(*x_prev) / detJ(*x_prev), x_prev[1] - detA2(*x_prev) / detJ(*x_prev)]

        if max([abs(x_cur[0] - x_prev[0]), abs(x_cur[1] - x_prev[1])]) < eps:
            print('Итераций: ', iteration)
            return x_cur
        x_prev = x_cur


def graph(equations):
    func1 = lambda x1, x2: eval(equations[0])
    func2 = lambda x1, x2: eval(equations[1])

    plt.figure()
    x1_list = linspace(0.1, 4.0, 2000)
    x2_list = linspace(0.1, 4.0, 2000)
    plt.xlabel('x1')
    plt.ylabel('x2')
    x1, x2 = meshgrid(x1_list, x2_list)
    plt.contour(x1, x2, func1(x1, x2), [0], colors='k')
    plt.contour(x1, x2, func2(x1, x2), [0], colors='r')
    plt.show()


def test(eq1, eq2, x_1, x_2):
    print('Проверка')
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')

    exp1 = parse_expr(eq1)
    exp2 = parse_expr(eq2)

    func1 = sympy.lambdify([x1, x2], exp1)
    func2 = sympy.lambdify([x1, x2], exp2)

    print('{0:.16f}'.format(func1(x_1, x_2)))
    print('{0:.16f}'.format(func2(x_1, x_2)))


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
        Equations = data['equations']
        Intervals = data['intervals']

        if method == 1:
            Fi = data['fi']

    graph(Equations)

    if method == 1:
        X = method_of_simple_iteration(*Fi, *Intervals, Eps)
    elif method == 2:
        X = newton_method(*Equations, *Intervals, Eps)

    for i in range(0, len(X)):
        print('x{0} = {1}'.format(i + 1, X[i]))
    test(*Equations, *X)
