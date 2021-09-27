import argparse
import json
import sympy
import matplotlib.pyplot as plt
from numpy import *
from sympy.parsing.sympy_parser import parse_expr
from utils import Matrix, Vector
from lu import lu_solve


def get_points(begin, end, step):
    return arange(begin, end, step).tolist() + [end]


def get_values(points, func):
    return [func(points[i]) for i in range(len(points))]


def get_matrix_system(points, h, lambda_, k):
    size_p = len(points)
    A = Matrix(size_p)

    for i in range(size_p):
        for j in range(size_p):
            if i == j:
                if j == 0 or j == size_p - 1:
                    A[i][j] = 1 - lambda_ * h / 2 * k(points[i], points[j])
                else:
                    A[i][j] = 1 - lambda_ * h * k(points[i], points[j])
            elif j == 0 or j == size_p - 1:
                A[i][j] = - lambda_ * h / 2 * k(points[i], points[j])
            else:
                A[i][j] = - lambda_ * h * k(points[i], points[j])
    return A


def quadratures_method(points, h, lambda_exp, k_exp, g_exp):
    x = sympy.Symbol('x')
    t = sympy.Symbol('t')

    k = sympy.lambdify([x, t], parse_expr(k_exp))
    g = sympy.lambdify(x, parse_expr(g_exp))
    lambda_ = float(parse_expr(lambda_exp))

    A = get_matrix_system(points, h, lambda_, k)
    G = Vector(len(points))
    G.vector = get_values(points, g)

    return lu_solve(A, G).vector


def get_error(points, result, sol):
    return sum([(result[i] - sol(points[i])) ** 2 for i in range(len(points))]) ** 0.5


def graph(points, values, h, sol=None):
    plt.subplots()
    if sol is not None:
        x = linspace(points[0], points[-1], 100000)
        plt.plot(x, sol(x), color='g', label='Аналитическое решение', linewidth=3, linestyle='--')

    plt.plot(points, values, color='r', label='Численное решение')
    if sol is not None:
        plt.title(f'Шаг = {h}, ошибка = {get_error(points, values, sol)}')
    else:
        plt.title(f'Шаг = {h}')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    solution = None
    with open(args.input, 'r') as f:
        data = json.load(f)
        k_exp = data['K']
        g_exp = data['g']
        lambda_exp = data['lambda']
        interval = data['interval']
        h = data['step']

        if 'solution' in data.keys():
            x = sympy.Symbol('x')
            solution = sympy.lambdify(x, parse_expr(data['solution']))

    points = get_points(*interval, h)

    result = quadratures_method(points, h, lambda_exp, k_exp, g_exp)
    graph(points, result, h, solution)


if __name__ == '__main__':
    main()
