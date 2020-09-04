import argparse
import copy
import json
import math

from utils import Matrix, Vector


def t(a):
    sum_el = 0.0
    for j in range(1, a.size):
        for i in range(0, j):
            sum_el += a[i][j] ** 2
    return math.sqrt(sum_el)


def find_max(a):
    i_max, j_max = 0, 1
    for j in range(1, a.size):
        for i in range(0, j):
            if abs(a[i][j]) > abs(a[i_max][j_max]):
                i_max, j_max = i, j
    return i_max, j_max


def method_jacobi_rotations(a, eps):
    n_iter = 0
    a_k = copy.copy(a)
    v = Matrix(a.size, single=True)

    while True:
        i_max, j_max = find_max(a_k)
        fi = 0.5 * math.atan(2 * a_k[i_max][j_max] / (a_k[i_max][i_max] - a_k[j_max][j_max]))

        u = Matrix(a.size, single=True)
        u[i_max][i_max] = math.cos(fi)
        u[i_max][j_max] = -math.sin(fi)
        u[j_max][i_max] = math.sin(fi)
        u[j_max][j_max] = math.cos(fi)

        u_t = copy.copy(u)
        u_t.transpose()

        a_k = u_t * a_k * u
        v = v * u

        n_iter += 1

        if t(a_k) < eps:
            eigenvalue = Vector(a_k.size)
            eigenvalue.vector = [a_k[i][i] for i in range(0, a_k.size)]

            print('Итераций: ', n_iter)
            return eigenvalue, v


def test(eigenvalue, v_matrix, matrix):
    print('Проверка')
    vectors = []

    for vector in zip(*v_matrix.matrix):
        v = Vector(v_matrix.size)
        v.vector = vector
        vectors.append(v)

        for i in range(0, len(vectors) - 1):
            print('v_{0}: '.format(i + 1), end='')
            vectors[i].print_vector()
            print('v_{0}: '.format(len(vectors)), end='')
            v.print_vector()
            print('(v_{0}, v_{1}): '.format(i + 1, len(vectors)), end='')
            res = sum([vectors[i][j] * v[j] for j in range(0, v.size)])
            print('{0:8.20f}'.format(res))
            print()

    print('Проверка A * x = a_k * x')

    for i in range(0, len(vectors)):
        print('a_k =  ', eigenvalue[i])
        print('x =  ', end='')
        vectors[i].print_vector()
        print('A * x =  ', end='')
        (matrix * vectors[i]).print_vector()
        print('a_k * x =  ', end='')
        (vectors[i] * eigenvalue[i]).print_vector()
        print('-------------------------------------------------------------')


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

    Eigenvalue, V = method_jacobi_rotations(A, Eps)

    print('Собственные значения: ')
    for i in range(0, n):
        print('a_{0} = {1:8.5f}'.format(i + 1, Eigenvalue[i]))
    print('\nМатрица собственных векторов: ')
    V.print_matrix()

    test(Eigenvalue, V, A)
