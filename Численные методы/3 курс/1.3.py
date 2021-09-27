import argparse
import copy
import json

from utils import Matrix, Vector


def get_alpha_beta(a, b):
    size = a.size
    alpha = Matrix(size)
    beta = Vector(size)

    for i in range(0, size):
        beta[i] = b[i] / a[i][i]
        for j in range(0, size):
            alpha[i][j] = -a[i][j] / a[i][i] if i != j else 0

    return alpha, beta


def method_of_simple_iteration(alpha, beta, eps):
    n_iter = 0
    x = copy.deepcopy(beta)
    alpha_norm = alpha.matrix_norm()
    k = alpha_norm / (1 - alpha_norm)

    while True:
        x_prev = copy.deepcopy(x)
        x = beta + alpha * x
        n_iter += 1
        if k * (x - x_prev).vector_norm() < eps:
            print('Итераций: ', n_iter)
            return x


def seidel_method(alpha, beta, eps):
    n_iter = 0
    x = copy.deepcopy(beta)
    alpha_norm = alpha.matrix_norm()
    k = alpha_norm / (1 - alpha_norm)

    while True:
        x_prev = copy.deepcopy(x)
        for i in range(0, x.size):
            x[i] = beta[i] + sum([alpha[i][j] * x[j] for j in range(0, x.size)])
        n_iter += 1
        if k * (x - x_prev).vector_norm() < eps:
            print('Итераций: ', n_iter)
            return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    while True:
        method = int(input('Метод простых итераций - 1\nМетод Зейделя - 2\n'))
        if method in (1, 2):
            break

    with open(args.input, 'r') as f:
        data = json.load(f)
        n = int(data['size'])
        Eps = float(data['eps'])

    A = Matrix(n)
    B = Vector(n)

    A.matrix_read_file(args.input, 'matrix')
    B.vector_read_file(args.input, 'vector')

    Alpha, Beta = get_alpha_beta(A, B)
    print('Альфа:')
    Alpha.print_matrix()
    print('Бета:')
    Beta.print_vector()
    print('Норма матрицы Альфа: ', Alpha.matrix_norm())
    print()

    X = Vector(n)
    if method == 1:
        X = method_of_simple_iteration(Alpha, Beta, Eps)
    if method == 2:
        X = seidel_method(Alpha, Beta, Eps)

    for i in range(0, X.size):
        print('x{0} = {1:8.5f}'.format(i + 1, X[i]))
