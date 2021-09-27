from numpy import *


def get_points(begin, end, step):
    return arange(begin, end, step).tolist() + [end]


def method_runge_kutta(points, start_cond_0, start_cond_1, step, f, g):
    values = [start_cond_0]
    values_der = [start_cond_1]

    for i in range(0, len(points) - 1):
        x_k = points[i]
        y_k = values[i]
        z_k = values_der[i]

        k_1 = step * f(x_k, y_k, z_k)
        l_1 = step * g(x_k, y_k, z_k)

        k_2 = step * f(x_k + 0.5 * step, y_k + 0.5 * k_1, z_k + 0.5 * l_1)
        l_2 = step * g(x_k + 0.5 * step, y_k + 0.5 * k_1, z_k + 0.5 * l_1)

        k_3 = step * f(x_k + 0.5 * step, y_k + 0.5 * k_2, z_k + 0.5 * l_2)
        l_3 = step * g(x_k + 0.5 * step, y_k + 0.5 * k_2, z_k + 0.5 * l_2)

        k_4 = step * f(x_k + step, y_k + k_3, z_k + l_3)
        l_4 = step * g(x_k + step, y_k + k_3, z_k + l_3)

        dy_k = 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        dz_k = 1 / 6 * (l_1 + 2 * l_2 + 2 * l_3 + l_4)

        values.append(y_k + dy_k)
        values_der.append(z_k + dz_k)

    return values, values_der
