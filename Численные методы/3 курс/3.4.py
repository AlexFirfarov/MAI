import argparse
import json


def find_interval(points, x_p):
    for i in range(0, len(points) - 2):
        if points[i] < x_p <= points[i + 1]:
            return i
    return -1


def get_derivative_2(points, val, x_p):
    i = find_interval(points, x_p)
    if i == -1:
        raise Exception("Недостаточно точек")

    k_1 = (val[i + 2] - val[i + 1]) / (points[i + 2] - points[i + 1])
    k_2 = ((val[i + 1] - val[i]) / (points[i + 1] - points[i]))

    return 2 * (k_1 - k_2) / (points[i + 2] - points[i])


def get_derivative_1(points, val, x_p):
    i = find_interval(points, x_p)
    if i == -1:
        raise Exception("Недостаточно точек")

    k_1 = (val[i + 1] - val[i]) / (points[i + 1] - points[i])
    k_2 = (val[i + 2] - val[i + 1]) / (points[i + 2] - points[i + 1])
    k_3 = (2 * x_p - points[i] - points[i + 1]) / (points[i + 2] - points[i])

    return k_1 + (k_2 - k_1) * k_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
        Points = data['points']
        Values = data['values']
        X_p = data['x_p']

    dev1 = get_derivative_1(Points, Values, X_p)
    dev2 = get_derivative_2(Points, Values, X_p)

    print('Точка x = ', X_p)
    print('Первая производная: ', dev1)
    print('Вторая производная: ', dev2)
