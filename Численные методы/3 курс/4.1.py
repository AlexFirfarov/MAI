import argparse
import json
import sympy
import matplotlib.pyplot as plt
from numpy import *
from sympy.parsing.sympy_parser import parse_expr


def exact_values(points, sol_f):
    return [sol_f(p) for p in points]


def method_runge_kutta(points, start_cond_0, start_cond_1, step, f, g):
    values = [start_cond_0]
    values_der = [start_cond_1]

    for i in range(0, len(points) - 1):
        x_k = points[i]
        y_k = values[i]
        z_k = values_der[i]

        k_1 = step * f(x_k, y_k, z_k)
        l_1 = step * g(x_k, y_k, z_k)

        k_2 = step * f(x_k + 0.5 * step, y_k + 0.5 * k_1, z_k + 0.5 * l_1)
        l_2 = step * g(x_k + 0.5 * step, y_k + 0.5 * k_1, z_k + 0.5 * l_1)

        k_3 = step * f(x_k + 0.5 * step, y_k + 0.5 * k_2, z_k + 0.5 * l_2)
        l_3 = step * g(x_k + 0.5 * step, y_k + 0.5 * k_2, z_k + 0.5 * l_2)

        k_4 = step * f(x_k + step, y_k + k_3, z_k + l_3)
        l_4 = step * g(x_k + step, y_k + k_3, z_k + l_3)

        dy_k = 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        dz_k = 1 / 6 * (l_1 + 2 * l_2 + 2 * l_3 + l_4)

        values.append(y_k + dy_k)
        values_der.append(z_k + dz_k)

    return values, values_der


def method_euler(points, start_cond_0, start_cond_1, step, f, g):
    values = [start_cond_0]
    values_der = [start_cond_1]

    for i in range(0, len(points) - 1):
        x_k = points[i]
        y_k = values[i]
        z_k = values_der[i]

        values.append(y_k + step * f(x_k, y_k, z_k))
        values_der.append(z_k + step * g(x_k, y_k, z_k))

    return values, values_der


def method_adams(x, start_cond_0, start_cond_1_, step, f, g):
    y, z = method_runge_kutta(x[:4], start_cond_0, start_cond_1_, step, f, g)

    for i in range(3, len(x) - 1):
        y.append(
            y[i] + step / 24 * (55 * f(x[i], y[i], z[i]) -
                                59 * f(x[i - 1], y[i - 1], z[i - 1]) +
                                37 * f(x[i - 2], y[i - 2], z[i - 2]) -
                                9 * f(x[i - 3], y[i - 3], z[i - 3]))
        )
        z.append(
            z[i] + step / 24 * (55 * g(x[i], y[i], z[i]) -
                                59 * g(x[i - 1], y[i - 1], z[i - 1]) +
                                37 * g(x[i - 2], y[i - 2], z[i - 2]) -
                                9 * g(x[i - 3], y[i - 3], z[i - 3]))
        )
    return y, z


def get_sol(points, coeffs, start_cond_0, start_cond_1, step, solution):
    f_exp = parse_expr('z')
    g_exp = parse_expr(f'-({coeffs[1]} * z + {coeffs[2]} * y + {coeffs[3]}) / ({coeffs[0]})')
    sol_exp = parse_expr(solution)

    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    f = sympy.lambdify([x, y, z], f_exp)
    g = sympy.lambdify([x, y, z], g_exp)
    sol_f = sympy.lambdify(x, sol_exp)

    val_euler, val_der_euler = method_euler(points, start_cond_0, start_cond_1, step, f, g)
    val_runge_kutta, val_der_runge_kutta = method_runge_kutta(points, start_cond_0, start_cond_1, step, f, g)
    val_adams, val_der_adams = method_adams(points, start_cond_0, start_cond_1, step, f, g)
    val_exact = exact_values(points, sol_f)

    return {'euler': val_euler, 'runge_kutta': val_runge_kutta, 'adams': val_adams, 'exact': val_exact}


def err_runge_romberg(res_1, res_2, p):
    return (sum([(res_1[i] - res_2[2 * i]) ** 2 for i in range(0, len(res_1))]) ** 0.5) / (2 ** p - 1)


def get_err_runge_romberg(res_1, res_2):
    err_euler = err_runge_romberg(res_1['euler'], res_2['euler'], 1)
    err_runge_kutta = err_runge_romberg(res_1['runge_kutta'], res_2['runge_kutta'], 4)
    err_adams = err_runge_romberg(res_1['adams'], res_2['adams'], 4)

    return {'euler': err_euler, 'runge_kutta': err_runge_kutta, 'adams': err_adams}


def err_exact(res, exc_val):
    return sum([(res[i] - exc_val[i]) ** 2 for i in range(0, len(res))]) ** 0.5


def get_err_exact(res):
    exc_val = res['exact']

    exc_err_euler = err_exact(res['euler'], exc_val)
    exc_err_runge_kutta = err_exact(res['runge_kutta'], exc_val)
    exc_err_adams = err_exact(res['adams'], exc_val)

    return {'euler': exc_err_euler, 'runge_kutta': exc_err_runge_kutta, 'adams': exc_err_adams}


def get_points(begin, end, step):
    return arange(begin, end, step).tolist() + [end]


def print_results(res, err_rr, err_exc, pt):
    print('{0:18s}'.format('Точки'), end='')
    for point in pt:
        print('{0:15.2f}'.format(point), end='')
    print('\n' + '-' * 184)

    print('{0:18s}'.format('Метод Эйлера'), end='')
    for val in res['euler']:
        print('{0:15.10f}'.format(val), end='')

    print('\n{0:18s}'.format('Метод Рунге-Кутты'), end='')
    for val in res['runge_kutta']:
        print('{0:15.10f}'.format(val), end='')

    print('\n{0:18s}'.format('Метод Адамса'), end='')
    for val in res['adams']:
        print('{0:15.10f}'.format(val), end='')

    print('\n{0:18s}'.format('Точное решение'), end='')
    for val in res['exact']:
        print('{0:15.10f}'.format(val), end='')

    print('\n' + '-' * 76 + 'Ошибки по методу  Рунге-Ромберга' + '-' * 76)
    print('{0:18s}'.format('Метод Эйлера'), end='')
    print('{0:15.12f}'.format(err_rr['euler']), end='')

    print('\n{0:18s}'.format('Метод Рунге-Кутты'), end='')
    print('{0:15.12f}'.format(err_rr['runge_kutta']), end='')

    print('\n{0:18s}'.format('Метод Адамса'), end='')
    print('{0:15.12f}'.format(err_rr['adams']), end='')

    print('\n' + '-' * 89 + 'Ошибки' + '-' * 89)
    print('{0:18s}'.format('Метод Эйлера'), end='')
    print('{0:15.12f}'.format(err_exc['euler']), end='')

    print('\n{0:18s}'.format('Метод Рунге-Кутты'), end='')
    print('{0:15.12f}'.format(err_exc['runge_kutta']), end='')

    print('\n{0:18s}'.format('Метод Адамса'), end='')
    print('{0:15.12f}'.format(err_exc['adams']), end='')


def graph(res, pt):
    plt.subplots()
    plt.plot(pt, res['euler'], color='r', label='Euler method')
    plt.plot(pt, res['runge_kutta'], color='b', label='Runge Kutta method')
    plt.plot(pt, res['adams'], color='g', label='Adams method')
    plt.plot(pt, res['exact'], color='y', label='Exact solution')
    for i in range(0, len(pt)):
        plt.scatter(pt[i], res['euler'][i], color='r')
        plt.scatter(pt[i], res['runge_kutta'][i], color='b')
        plt.scatter(pt[i], res['adams'][i], color='g')
        plt.scatter(pt[i], res['exact'][i], color='y')
    plt.legend()
    plt.title('Решения')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
        Coefficients = data['coefficients']
        Start_cond_0 = data['condition_0']
        Start_cond_1 = data['condition_1']
        Interval = data['interval']
        Step = data['step']
        Solution = data['solution']

    points = get_points(*Interval, Step)
    points_2 = get_points(*Interval, Step / 2)

    coeffs = [parse_expr(coeff) for coeff in Coefficients]
    cond_0 = float(parse_expr(Start_cond_0))
    cond_1 = float(parse_expr(Start_cond_1))

    result = get_sol(points, coeffs, cond_0, cond_1, Step, Solution)
    result_2 = get_sol(points_2, coeffs, cond_0, cond_1, Step / 2, Solution)

    err_runge_romberg = get_err_runge_romberg(result, result_2)
    err_exact = get_err_exact(result)

    graph(result, points)
    print_results(result, err_runge_romberg, err_exact, points)
