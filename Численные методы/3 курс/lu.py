import copy
from utils import Matrix, Vector


def lu(a):
    size = a.size

    l = Matrix(size, single=True)
    u = copy.deepcopy(a)

    for k in range(0, size - 1):
        m_k = Matrix(size, single=True)
        for i in range(k + 1, size):
            mu_i = -u[i][k] / u[k][k]

            m_k[i][k] = mu_i
            l[i][k] = -mu_i
        u = m_k * u
    return l, u


def lu_solve(a, b):
    l, u = lu(a)

    size = b.size
    z = Vector(size)
    x = Vector(size)

    z[0] = b[0]
    for i in range(1, size):
        z[i] = b[i] - sum([l[i][j] * z[j] for j in range(0, i)])

    x[size - 1] = z[size - 1] / u[size - 1][size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = (z[i] - sum([u[i][j] * x[j] for j in range(i + 1, size)])) / u[i][i]

    return x
