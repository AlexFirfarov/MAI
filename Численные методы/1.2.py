import json
import argparse
import sys


class TridiagonalMatrix:

    def __init__(self, size):
        self.A = []
        self.B = []
        self.C = []
        self.size = size

    def trimatrix_read_file(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if len(data['A']) != self.size - 1 or len(data['C']) != self.size - 1 or len(data['B']) != self.size:
                    raise ValueError
            self.A = [0.0] + list(map(float, data['A']))
            self.B = list(map(float, data['B']))
            self.C = list(map(float, data['C'])) + [0.0]
        except Exception:
            print('Некоректные входные данные')
            sys.exit(1)


class Vector:

    def __init__(self, size):
        self.size = size
        self.vector = [0.0 for _ in range(0, size)]

    def __getitem__(self, idx):
        return self.vector[idx]

    def __setitem__(self, key, value):
        self.vector[key] = value

    def print_vector(self):
        for num in self.vector:
            print('{0:5.2f}  '.format(num),  end='')
        print()

    def vector_read_file(self, filename, arg_name):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if len(data[arg_name]) != self.size:
                    raise ValueError
            self.vector = list(map(float, data[arg_name]))
        except Exception:
            print('Некоректные входные данные')
            sys.exit(1)


def rtm(m, d):
    size = m.size
    x = [0 for _ in range(0, size)]
    p, q = [], []

    p.append(-m.C[0] / m.B[0])
    q.append(d[0] / m.B[0])

    for i in range(1, size):
        p_i = -m.C[i] / (m.B[i] + m.A[i] * p[i - 1])
        q_i = (d[i] - m.A[i] * q[i - 1]) / (m.B[i] + m.A[i] * p[i - 1])

        p.append(p_i)
        q.append(q_i)

    x[size - 1] = q[size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    n = int(input('Размерность матрицы: '))
    M = TridiagonalMatrix(n)
    D = Vector(n)

    M.trimatrix_read_file(args.input)
    D.vector_read_file(args.input, 'D')

    X = rtm(M, D)
    for i in range(0, len(X)):
        print('x{0} = {1:5.2f}'.format(i + 1, X[i]))
