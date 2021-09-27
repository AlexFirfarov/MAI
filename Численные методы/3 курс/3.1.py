import argparse
import json
import sympy
import operator
import matplotlib.pyplot as plt
from numpy import *
from pprint import pprint
from functools import reduce
from sympy.parsing.sympy_parser import parse_expr


def get_w(points, idx):
    w = ''
    for i in range(0, len(points)):
        w += '(x {0} {1}) * '.format('-' if points[i] > 0 else '+', abs(points[i])) if i != idx else ''
    return w[0:-2]


def lagrange_interpolation(func, points):
    size = len(points)

    lagrange = ''
    for i in range(0, size):
        w_n = get_w(points, i)
        w_i = reduce(operator.mul, [points[i] - points[j] for j in range(0, size) if i != j])
        k = func(points[i]) / w_i
        if k > 0:
            lagrange += '+ '
        lagrange += '{0} * {1}'.format(k, w_n)

    if lagrange[0] == '+':
        lagrange = lagrange[1:]
    return lagrange


def newton_interpolation(func, points):
    size = len(points)
    divided_div = [func(point) for point in points]

    newton = '{0}'.format(divided_div[0])
    k = ''

    for i in range(1, size):
        for j in range(size - 1, i - 1, -1):
            divided_div[j] = (divided_div[j - 1] - divided_div[j]) / (points[j - i] - points[j])
        k += '(x {0} {1}) '.format('-' if points[i - 1] > 0 else '+', abs(points[i - 1]))
        if divided_div[i] > 0:
            newton += '+'
        newton += '{0} * {1}'.format(divided_div[i], k)
        k += '* '

    if newton[0] == '+':
        newton = newton[1:]
    return newton


def graph(f_poly, func):

    plt.subplots()
    x = linspace(-0.6, 0.6, 100000)
    plt.plot(x, f_poly(x), color='r', label='Полином')
    plt.plot(x, func(x), color='b', label='Точная функция')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    while True:
        method = int(input('Лагранж - 1\nНьютон - 2\n'))
        if method in (1, 2):
            break

    with open(args.input, 'r') as f:
        data = json.load(f)
        Function = data['function']
        Points = data['points']
        X_p = data['x_p']

    x = sympy.Symbol('x')
    exp = parse_expr(Function)
    function = sympy.lambdify(x, exp)

    for points_set in Points:
        if method == 1:
            poly = lagrange_interpolation(function, points_set)
        elif method == 2:
            poly = newton_interpolation(function, points_set)

        x = sympy.Symbol('x')
        exp_poly = sympy.expand(parse_expr(poly))
        func_poly = sympy.lambdify(x, exp_poly)

        print('Точки: ', points_set)
        print('Многочлен {0}: '.format('Лагранжа' if method == 1 else 'Ньютона'))
        pprint(poly, compact=True)
        print('Раскрытый вид: ')
        pprint(exp_poly)
        print('Абсолютная погрешность в точке {0}: '.format(X_p))

        print('{0:.16f}'.format(abs(func_poly(X_p) - function(X_p))))
        print('----------------------------------------------------')

        graph(func_poly, function)
