import json
import sys


class Matrix:

    def __init__(self, size, single=False):
        self.size = size
        self.matrix = []
        for _ in range(0, size):
            self.matrix.append([0.0 for _ in range(0, size)])

        if single:
            for i in range(0, size):
                for j in range(0, size):
                    if i == j:
                        self.matrix[i][j] = 1.0

    def __getitem__(self, idx):
        return self.matrix[idx]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __sub__(self, other):
        if isinstance(other, Matrix) and self.size == other.size:
            result = Matrix(self.size)
            for i in range(0, self.size):
                for j in range(0, self.size):
                    result[i][j] = self[i][j] - other[i][j]
            return result
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, Matrix) and self.size == other.size:
            result = Matrix(self.size)
            for i in range(0, self.size):
                for j in range(0, self.size):
                    result[i][j] = sum([self[i][k] * other[k][j] for k in range(0, self.size)])
            return result
        elif isinstance(other, Vector) and self.size == other.size:
            result = Vector(self.size)
            for i in range(0, self.size):
                result[i] = sum([self[i][j] * other[j] for j in range(0, self.size)])
            return result
        elif isinstance(other, float) or isinstance(other, int):
            result = Matrix(self.size)
            for i in range(0, self.size):
                for j in range(0, self.size):
                    result[i][j] = self[i][j] * other
            return result
        else:
            raise TypeError

    def print_matrix(self):
        for i in range(0, self.size):
            for j in range(0, self.size):
                print('{0:9.5f}'.format(self.matrix[i][j]), end=' ')
            print()
        print()

    def transpose(self):
        self.matrix = [list(i) for i in zip(*self.matrix)]

    def matrix_norm(self):
        return max([sum([abs(self.matrix[i][j]) for j in range(0, self.size)]) for i in range(0, self.size)])

    def matrix_read_file(self, filename, arg_name):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                matrix = data[arg_name]
                if len(matrix) != self.size:
                    raise ValueError
                for row in matrix:
                    if len(row) != self.size:
                        raise ValueError
                for i in range(0, len(matrix)):
                    matrix[i] = list(map(float, matrix[i]))
                self.matrix = matrix
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

    def __add__(self, other):
        if isinstance(other, Vector) and self.size == other.size:
            result = Vector(self.size)
            result.vector = [self[i] + other[i] for i in range(0, self.size)]
            return result
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, Vector) and self.size == other.size:
            result = Vector(self.size)
            result.vector = [self[i] - other[i] for i in range(0, self.size)]
            return result
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            result = Vector(self.size)
            for i in range(0, self.size):
                result[i] = self[i] * other
            return result
        else:
            raise TypeError

    def print_vector(self):
        for num in self.vector:
            print('{0:9.5f}  '.format(num),  end='')
        print()

    def vector_norm(self):
        return max([abs(self.vector[i]) for i in range(0, self.size)])

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
