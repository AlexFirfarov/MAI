import argparse
import copy
import json
import math
import numpy

from utils import Matrix, Vector
from numpy.linalg import eig


def get_matrix_householder(matrix, col):
    v = Vector(matrix.size)
    e = Matrix(matrix.size, single=True)

    sign = -1 if matrix[col][col] < 0 else 1 if matrix[col][col] > 0 else 0
    v[col] = matrix[col][col] + sign * math.sqrt(sum([matrix[j][col] ** 2 for j in range(col, matrix.size)]))

    for i in range(col + 1, matrix.size):
        v[i] = matrix[i][col]

    v_vt = Matrix(matrix.size)
    for i in range(0, v_vt.size):
        for j in range(0, v_vt.size):
            v_vt[i][j] = v[i] * v[j]

    vt_v = sum([v[i] ** 2 for i in range(0, v.size)])

    h = e - v_vt * (2 / vt_v)
    return h


def qr(matrix):
    q = Matrix(matrix.size, single=True)
    r = copy.copy(matrix)

    for i in range(0, matrix.size - 1):
        h = get_matrix_householder(r, i)

        q = q * h
        r = h * r

    return q, r


def qr_eigenvalues(matrix, eps):
    size = matrix.size
    a_k = copy.deepcopy(matrix)

    res = [None for _ in range(0, size)]
    iteration = 0

    while True:
        break_flag = True
        iteration += 1
        q_k, r_k = qr(a_k)
        a_k = r_k * q_k

        i = 0
        while i < size - 1:
            if math.sqrt(sum([a_k[j][i] ** 2 for j in range(i + 1, size)])) < eps:
                res[i] = a_k[i][i]
                i += 1
            else:
                roots = numpy.roots([1, -a_k[i + 1][i + 1] - a_k[i][i],
                                     a_k[i][i] * a_k[i + 1][i + 1] - a_k[i][i + 1] * a_k[i + 1][i]]).tolist()
                if not (isinstance(res[i], list) and \
                        abs(roots[0] - res[i][0]) < eps and \
                        abs(roots[1] - res[i][1]) < eps):
                    break_flag = False
                res[i] = copy.copy(roots)
                res[i + 1] = None
                i += 2
            if not break_flag:
                break

        if break_flag:
            answer = []
            print('Итераций: ', iteration)
            for val in res:
                if val is None:
                    continue
                elif isinstance(val, list):
                    answer.append(val[0])
                    answer.append(val[1])
                else:
                    answer.append(val)
            if len(answer) == size - 1:
                answer.append(a_k[size - 1][size - 1])
            return answer


def test(matrix):
    print()
    values, vectors = eig(matrix.matrix)
    print(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
        n = int(data['size'])
        Eps = float(data['eps'])

    A = Matrix(n)
    A.matrix_read_file(args.input, 'matrix')

    result = qr_eigenvalues(A, Eps)

    print('Собственные значения: ')
    for i in range(0, n):
        print('a_{0} = {1}'.format(i + 1, result[i]))

    test(A)
