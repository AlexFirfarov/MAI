import argparse
import json
import sympy
from numpy import arange
from sympy.parsing.sympy_parser import parse_expr


def get_points(x_0, x_k, step):
    return arange(x_0, x_k, step).tolist() + [x_k]


def rectangle_method(func, x, step):
    return step * sum([func((x[i] + x[i + 1]) / 2) for i in range(0, len(x) - 1)])


def trapeze_method(func, x, step):
    size = len(x)
    return step * (func(x[0]) / 2 + sum([func(x[i]) for i in range(1, size - 1)]) + func(x[size - 1]) / 2)


def simpson_method(func, x, step):
    size = len(x)
    return step / 3 * (func(x[0]) + sum([4 * func(x[i]) for i in range(1, size - 1, 2)]) +
                       sum([2 * func(x[i]) for i in range(2, size - 2, 2)]) + func(x[size - 1]))


def runge_romberg_method(res):
    k = res[0]['H'] / res[1]['H']

    r_rec = abs((res[1]['Rectangle'] - res[0]['Rectangle'])) / (k ** 2 - 1)
    r_tr = abs((res[1]['Trapeze'] - res[0]['Trapeze'])) / (k ** 2 - 1)
    r_simp = abs((res[1]['Simpson'] - res[0]['Simpson'])) / (k ** 4 - 1)

    return {'r_rec': r_rec, 'r_tr': r_tr, 'r_simp': r_simp}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
        Function = data['function']
        X_0 = data['x_0']
        X_k = data['x_k']
        Steps = data['steps']

    x = sympy.Symbol('x')
    exp = parse_expr(Function)
    function = sympy.lambdify(x, exp)

    results = []
    for h in Steps:
        X = get_points(X_0, X_k, h)

        res_rectangle = rectangle_method(function, X, h)
        res_trapeze = trapeze_method(function, X, h)
        res_simpson = simpson_method(function, X, h)

        print('Шаг: ', h)
        print('Метод прямоугольников: ', res_rectangle)
        print('Метод трапеций: ', res_trapeze)
        print('Метод Симпсона: ', res_simpson)
        print()

        results.append({
            'Rectangle': res_rectangle,
            'Trapeze': res_trapeze,
            'Simpson': res_simpson,
            'H': h
        })
    r_result = runge_romberg_method(results)

    print('Ошибки по методу Рунге-Ромберга'.format(Steps[0]))
    print('Метод прямоугольников: ', r_result['r_rec'])
    print('Метод трапеций: ', r_result['r_tr'])
    print('Метод Симпсона: ', r_result['r_simp'])
