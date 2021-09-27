import argparse
import json
import sympy
import matplotlib.pyplot as plt
from numpy import *
from sympy.parsing.sympy_parser import parse_expr
from utils import TridiagonalMatrix, Vector
from runge_kutta import method_runge_kutta, get_points
from rtm import rtm


def get_n(n_prev, n, res_prev, res_der_prev, res, res_der, delta, gamma, cond_1):
    y_der = res_der_prev[-1]
    y = (cond_1 - gamma * y_der) / delta
    phi_prev = y - res_prev[-1]

    y_der = res_der[-1]
    y = (cond_1 - gamma * y_der) / delta
    phi = y - res[-1]

    return n - phi * (n - n_prev) / (phi - phi_prev)


def check_end(res, res_der, delta, gamma, cond_1, eps):
    y_der = res_der[-1]
    y = (cond_1 - gamma * y_der) / delta
    phi = y - res[-1]
    return abs(phi) < eps


def finite_difference(pt, coeffs, alpha, beta, delta, gamma, step, cond_0, cond_1):
    x = sympy.Symbol('x')
    p_exp = parse_expr(f'({coeffs[1]}) / ({coeffs[0]})')
    q_exp = parse_expr(f'({coeffs[2]}) / ({coeffs[0]})')

    p = sympy.lambdify(x, p_exp)
    q = sympy.lambdify(x, q_exp)

    sz = len(pt) - 1
    a = [0.0] + [1 - p(pt[i]) * step / 2 for i in range(0, sz - 1)] + [- gamma]
    b = [alpha * step - beta] + [-2 + q(pt[i]) * step ** 2 for i in range(0, sz - 1)] + [delta * step + gamma]
    c = [beta] + [1 + p(pt[i]) * step / 2 for i in range(0, sz - 1)] + [0.0]
    d = [cond_0 * step] + [0 for _ in range(0, sz - 1)] + [cond_1 * step]

    matr = TridiagonalMatrix(sz + 1)
    matr.A = a
    matr.B = b
    matr.C = c

    vec = Vector(sz + 1)
    vec.vector = d

    return rtm(matr, vec)


def shooting_method(points, coeffs, alpha, beta, delta, gamma, cond_0, cond_1, step, eps):
    f_exp = parse_expr('z')
    g_exp = parse_expr(f'-({coeffs[1]} * z + {coeffs[2]} * y + {coeffs[3]}) / ({coeffs[0]})')

    if coeffs[0] == 'x' and points[0] == 0.0:
        points[0] += eps

    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    f = sympy.lambdify([x, y, z], f_exp)
    g = sympy.lambdify([x, y, z], g_exp)

    n_prev, n = 1.0, 0.8

    y_der = (cond_0 - alpha * n_prev) / beta
    res_prev, res_der_prev = method_runge_kutta(points, n_prev, y_der, step, f, g)

    y_der = (cond_0 - alpha * n) / beta
    res, res_der = method_runge_kutta(points, n, y_der, step, f, g)

    while not check_end(res, res_der, delta, gamma, cond_1, eps):
        n_prev, n = n, get_n(n_prev, n, res_prev, res_der_prev, res, res_der, delta, gamma, cond_1)
        res_prev = res
        y_der = (cond_0 - alpha * n) / beta
        res, res_der = method_runge_kutta(points, n, y_der, step, f, g)
    return res


def get_err_rr(res_1, res_2):
    return (sum([(res_1[i] - res_2[2 * i]) ** 2 for i in range(0, len(res_1))]) ** 0.5) / (2 ** 1 - 1)


def get_err(res, val):
    return sum([(res[i] - val[i]) ** 2 for i in range(0, len(res))]) ** 0.5


def print_results(shooting, fin_diff, exact, err_rr_d, err_d, points):
    print('{0:25s}'.format('Точки'), end='')
    for point in points:
        print('{0:10.2f}'.format(point), end='')
    print('\n' + '-' * 136)

    print('{0:25s}'.format('Метод стрельбы'), end='')
    for val in shooting:
        print('{0:10.5f}'.format(val), end='')

    print('\n{0:25s}'.format('Конечно разностный метод'), end='')
    for val in fin_diff:
        print('{0:10.5f}'.format(val), end='')

    print('\n{0:25s}'.format('Точное решение'), end='')
    for val in exact:
        print('{0:10.5f}'.format(val), end='')

    print('\n' + '-' * 52 + 'Ошибки по методу  Рунге-Ромберга' + '-' * 52)
    print('{0:25s}'.format('Метод стрельбы'), end='')
    print('{0:15.12f}'.format(err_rr_d['shooting']), end='')

    print('\n{0:25s}'.format('Конечно разностный метод'), end='')
    print('{0:15.12f}'.format(err_rr_d['finite_difference']), end='')

    print('\n' + '-' * 65 + 'Ошибки' + '-' * 65)
    print('{0:25s}'.format('Метод стрельбы'), end='')
    print('{0:15.12f}'.format(err_d['shooting']), end='')

    print('\n{0:25s}'.format('Конечно разностный метод'), end='')
    print('{0:15.12f}'.format(err_d['finite_difference']), end='')


def graph(points, val_shooting, val_finite_difference, f, step):
    plt.subplots()
    x = linspace(0, 1, 100000)
    plt.plot(x, f(x), color='r', label='Аналитическое решение')
    plt.plot(points, val_shooting, color='b', label='Метод стрельбы')
    plt.plot(points, val_finite_difference, color='g', label='Конечно разностный метод')
    plt.title(f'Решение с шагом {step}')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
        Coefficients = data['coefficients']
        Alpha, Beta = data['coefficients_1']
        Delta, Gamma = data['coefficients_2']
        Cond_0 = data['condition_0']
        Cond_1 = data['condition_1']
        Interval = data['interval']
        H = data['step']
        Eps = data['eps']
        Solution = data['solution']

    Cond_0 = float(parse_expr(Cond_0))
    Cond_1 = float(parse_expr(Cond_1))

    x = sympy.Symbol('x')
    f_exp = parse_expr(Solution)
    f = sympy.lambdify(x, f_exp)

    Points = get_points(*Interval, H)
    Points2 = get_points(*Interval, H / 2)

    Exact = []
    for Point in Points:
        Exact.append(f(Point))

    res_shooting = shooting_method(Points, Coefficients, Alpha, Beta, Delta, Gamma, Cond_0, Cond_1, H, Eps)
    res_finite_difference = finite_difference(Points, Coefficients, Alpha, Beta, Delta, Gamma, H, Cond_0, Cond_1)

    res_shooting2 = shooting_method(Points2, Coefficients, Alpha, Beta, Delta, Gamma, Cond_0, Cond_1, H / 2, Eps)
    res_finite_difference2 = finite_difference(Points2, Coefficients, Alpha, Beta, Delta, Gamma, H / 2, Cond_0, Cond_1)

    err_rr_shooting = get_err_rr(res_shooting, res_shooting2)
    err_rr_finite_difference = get_err_rr(res_finite_difference, res_finite_difference2)

    err_shooting = get_err(res_shooting, Exact)
    err_finite_difference = get_err(res_finite_difference, Exact)

    err_rr = {'shooting': err_rr_shooting, 'finite_difference': err_rr_finite_difference}
    err = {'shooting': err_shooting, 'finite_difference': err_finite_difference}

    graph(Points, res_shooting, res_finite_difference, f, H)
    print_results(res_shooting, res_finite_difference, Exact, err_rr, err, Points)
