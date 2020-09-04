import json
import argparse
import sys
import copy

import scipy.linalg
import scipy
from pprint import pprint

from utils import Matrix, Vector


def get_matrix_permutations(matrix):
    size = matrix.size
    p = Matrix(size, single=True)
    for i in range(0, size):
        column = [matrix[j][i] for j in range(i, size)]
        row_idx = column.index(max(column, key=abs)) + i
        if i != row_idx:
            p[i], p[row_idx] = p[row_idx], p[i]
    return p


def lup(matrix):
    size = matrix.size

    p = get_matrix_permutations(matrix)
    pa = p * matrix
    l = Matrix(size, single=True)
    u = copy.deepcopy(pa)

    for k in range(0, size - 1):
        m_k = Matrix(size, single=True)
        for i in range(k + 1, size):
            mu_i = -u[i][k] / u[k][k]

            m_k[i][k] = mu_i
            l[i][k] = -mu_i
        u = m_k * u
    return l, u, p


def lup_solve(l, u, p, b):
    size = b.size
    n_b = p * b
    z = Vector(size)
    x = Vector(size)

    z[0] = n_b[0]
    for i in range(1, size):
        z[i] = n_b[i] - sum([l[i][j] * z[j] for j in range(0, i)])

    x[size - 1] = z[size - 1] / u[size - 1][size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = (z[i] - sum([u[i][j] * x[j] for j in range(i + 1, size)])) / u[i][i]

    return x


def lup_det(u, p):
    det = 1.0
    count_perm = 0

    for i in range(0, p.size):
        if p[i][i] == 0:
            count_perm += 1
    count_perm = count_perm - 1 if count_perm > 0 else 0

    det *= (-1) ** count_perm
    for i in range(0, u.size):
        det *= u[i][i]
    return det


def inverse_matrix(l, u, p):
    size = l.size
    inv = Matrix(size)

    for i in range(0, size):
        e = Vector(size)
        e[i] = 1
        column = lup_solve(l, u, p, e)

        for j in range(0, size):
            inv[j][i] = column[j]
    return inv


def test(l, u, p, matrix, b, x, inv):
    print('----------------Проверка----------------')
    lu = l * u
    print('L * U: ')
    lu.print_matrix()
    pa = p * matrix
    print('P * A: ')
    pa.print_matrix()
    ax = matrix * x
    print('Ax: ')
    ax.print_vector()
    print('B: ')
    b.print_vector()
    print('A * A^-1: ')
    e = matrix * inv
    e.print_matrix()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    n = int(input('Размерность матрицы: '))
    A = Matrix(n)
    B = Vector(n)

    A.matrix_read_file(args.input, 'matrix')
    B.vector_read_file(args.input, 'vector')

    L, U, P = lup(A)

    L.print_matrix()
    U.print_matrix()
    P.print_matrix()
    print('-----------------------------------')

    P, L, U = scipy.linalg.lu(A.matrix)
    pprint(L)
    pprint(U)
    pprint(P)

    exit(0)

    X = lup_solve(L, U, P, B)
    det = lup_det(U, P)
    inv = inverse_matrix(L, U, P)

    for i in range(0, X.size):
        print('x{0} = {1:5.2f}'.format(i + 1, X[i]))

    print('\nОпределитель: {0:5.2f}'.format(det))
    print('\nОбратная матрица ')
    inv.print_matrix()

    #test(L, U, P, A, B, X, inv)
