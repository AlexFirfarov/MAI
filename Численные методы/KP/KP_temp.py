
def simpson(points, h, lambda_, k):
    print('simpson')
    size_p = len(points)
    A = Matrix(size_p)

    for i in range(size_p):
        for j in range(size_p):
            if i == j:
                if j == 0 or j == size_p - 1:
                    A[i][j] = 1 - lambda_ * h / 3 * k(points[i], points[j])
                elif j % 2:
                    A[i][j] = 1 - lambda_ * 4 * h / 3 * k(points[i], points[j])
                else:
                    A[i][j] = 1 - lambda_ * 2 * h / 3 * k(points[i], points[j])
            else:
                if j == 0 or j == size_p:
                    A[i][j] = - lambda_ * h / 3 * k(points[i], points[j])
                elif j % 2:
                    A[i][j] = - lambda_ * 4 * h / 3 * k(points[i], points[j])
                else:
                    A[i][j] = - lambda_ * 2 * h / 3 * k(points[i], points[j])
    return A.matrix


def get_alpha_beta(a, b, eps):
    size_matrix = a.size
    alpha = Matrix(size_matrix)
    beta = Vector(size_matrix)

    for i in range(size_matrix):
        beta[i] = b[i] / a[i][i]
        for j in range(size_matrix):
            alpha[i][j] = -a[i][j] / a[i][i] if i != j else 0
    return alpha, beta


def seidel_method(alpha, beta, eps):
    res = copy.copy(beta)
    alpha_norm = alpha.matrix_norm()
    if alpha_norm < 1:
        k = alpha_norm / (1 - alpha_norm)
    else:
        k = 1

    while True:
        res_prev = copy.copy(res)
        for i in range(0, res.size):
            res[i] = beta[i] + sum([alpha[i][j] * res[j] for j in range(0, res.size)])
        if k * (res - res_prev).vector_norm() < eps:
            return res.vector
