import argparse
import json
import sympy
import matplotlib.pyplot as plt
from utils import TridiagonalMatrix, Vector
from rtm import rtm
from numpy import *
from sympy.parsing.sympy_parser import parse_expr


def find_interval(points, x):
    for i in range(1, len(points)):
        if points[i - 1] < x <= points[i]:
            return i


def get_c(x, f, h):
    a_diag = [0.0] + [h[i - 1] for i in range(3, len(x))]
    b_diag = [2 * (h[i - 1] + h[i]) for i in range(2, len(x))]
    c_diag = [h[i] for i in range(2, len(x) - 1)] + [0.0]
    d = [3 * ((f[i] - f[i - 1]) / h[i] - ((f[i - 1] - f[i - 2]) / h[i - 1])) for i in range(2, len(x))]

    matrix = TridiagonalMatrix(len(x) - 2)
    matrix.A = a_diag
    matrix.B = b_diag
    matrix.C = c_diag

    vec = Vector(len(x) - 2)
    vec.vector = d

    c = [0.0, 0.0] + rtm(matrix, vec)
    return c


def get_a(f):
    return [0.0] + [f[i - 1] for i in range(1, len(f))]


def get_b(f, h, c):
    last = len(f) - 1
    b = [0.0]
    b.extend([(f[i] - f[i - 1]) / h[i] - 1 / 3 * h[i] * (c[i + 1] + 2 * c[i]) for i in range(1, last)])
    b.append((f[last] - f[last - 1]) / h[last] - 2 / 3 * h[last] * c[last])
    return b


def get_d(h, c):
    last = len(h) - 1
    d = [0.0]
    d.extend([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(1, last)])
    d.append(-c[last] / (3 * h[last]))
    return d


def spline_interpolation(x, f):
    h = [0.0] + [x[i] - x[i - 1] for i in range(1, len(x))]
    c = get_c(x, f, h)
    a = get_a(f)
    b = get_b(f, h, c)
    d = get_d(h, c)

    return a, b, c, d


def make_splines_exp(a, b, c, d, pt):
    splines_exp = []
    for i in range(1, len(pt)):
        splines_exp.append(
            f'{a[i]} + {b[i]} * (x - {pt[i - 1]}) + {c[i]} * (x - {pt[i - 1]})**2 + {d[i]} * (x - {pt[i - 1]})**3')
    return splines_exp


def print_splines(splines_exp, points):
    for i in range(0, len(points) - 1):
        print('Отрезок [{0},{1}]'.format(points[i], points[i + 1]))
        print(sympy.expand(parse_expr(splines_exp[i])))
    print()


def make_splines_func(splines_exp):
    x = sympy.Symbol('x')
    splines_func = []
    for spline_exp in splines_exp:
        splines_func.append(sympy.lambdify(x, spline_exp))
    return splines_func


def graph(splines_f, points):
    colors = ['red', 'blue', 'orange', 'green']
    size = len(points)

    plt.subplots()
    for i in range(0, size - 1):
        x = linspace(points[i], points[i + 1], 1000000)
        plt.plot(x, splines_f[i](x), color=colors[i])
        plt.scatter(points[i], splines_f[i](points[i]))
    plt.scatter(points[size - 1], splines_f[i](points[size - 1]))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as file:
        data = json.load(file)
        Points = data['points']
        Values = data['values']
        X_p = data['x_p']

    A, B, C, D = spline_interpolation(Points, Values)

    Splines = make_splines_exp(A, B, C, D, Points)
    Splines_f = make_splines_func(Splines)
    print_splines(Splines, Points)
    graph(Splines_f, Points)

    idx = find_interval(Points, X_p)
    print('Значение в точке {0}:'.format(X_p))
    print(Splines_f[idx](X_p))
