import argparse
import json
import sympy
import matplotlib.pyplot as plt
from utils import Matrix, Vector
from lup import lup, lup_solve
from numpy import *
from sympy.parsing.sympy_parser import parse_expr


def get_poly_1_deg(points, values):
    sum_points = sum(points)
    sum_points_2 = sum([point ** 2 for point in points])
    sum_values = sum(values)
    sum_v_p = sum([values[i] * points[i] for i in range(0, len(points))])

    a = Matrix(2)
    a.matrix = [[len(points), sum_points], [sum_points, sum_points_2]]
    b = Vector(2)
    b.vector = [sum_values, sum_v_p]

    l, u, p = lup(a)
    coefficients = lup_solve(l, u, p, b).vector

    return parse_expr(f'{coefficients[0]} + {coefficients[1]} * x')


def get_poly_2_deg(points, values):
    sum_points = sum(points)
    sum_points_2 = sum([point ** 2 for point in points])
    sum_points_3 = sum([point ** 3 for point in points])
    sum_points_4 = sum([point ** 4 for point in points])
    sum_values = sum(values)
    sum_v_p = sum([values[i] * points[i] for i in range(0, len(points))])
    sum_v_p2 = sum([values[i] * points[i] ** 2 for i in range(0, len(points))])

    a = Matrix(3)
    a.matrix = [
        [len(points), sum_points, sum_points_2],
        [sum_points, sum_points_2, sum_points_3],
        [sum_points_2, sum_points_3, sum_points_4]]
    b = Vector(3)
    b.vector = [sum_values, sum_v_p, sum_v_p2]

    l, u, p = lup(a)
    coefficients = lup_solve(l, u, p, b).vector

    return parse_expr(f'{coefficients[0]} + {coefficients[1]} * x + {coefficients[2]} * x ** 2')


def get_poly_f(poly_exp):
    x = sympy.Symbol('x')
    return sympy.lambdify(x, poly_exp)


def get_sum_err_squares(poly_f, points, values):
    return sum([(poly_f(points[i]) - values[i]) ** 2 for i in range(0, len(points))])


def graph(poly_1_f, poly_2_f, points, values):
    size = len(points)

    plt.subplots()
    x = linspace(points[0], points[size - 1], 100000)
    plt.plot(x, poly_1_f(x), color='r', label='1-ой степени')
    plt.plot(x, poly_2_f(x), color='b', label='2-ой степени')
    for i in range(0, size - 1):
        plt.scatter(points[i], values[i], color='g')
    plt.scatter(points[size - 1], values[size - 1], color='g', label='Заданные точки')
    plt.legend()
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

    poly_1 = get_poly_1_deg(Points, Values)
    poly_2 = get_poly_2_deg(Points, Values)

    poly_f_1 = get_poly_f(poly_1)
    poly_f_2 = get_poly_f(poly_2)

    poly_l = [poly_1, poly_2]
    poly_l_f = [poly_f_1, poly_f_2]
    for i in range(2):
        print(f'Многочлен степени {i + 1}: ')
        print(poly_l[i])
        print(f'Сумма квадратов ошибок: {get_sum_err_squares(poly_l_f[i], Points, Values)}')
        print()
    graph(*poly_l_f, Points, Values)
