import copy
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
