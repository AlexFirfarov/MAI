#include <iostream>
#include <cmath>
#include <cstdlib>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return std::fabs(a) < std::fabs(b);
	}
};

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__global__ void kernel_swap_rows(double *matrix, int row_idx_1, int row_idx_2, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int i;
	double tmp;

	for (i = idx; i < m; i += offsetx) {
		tmp = matrix[i * n + row_idx_1];
		matrix[i * n + row_idx_1] = matrix[i * n + row_idx_2];
		matrix[i * n + row_idx_2] = tmp;
	}
}

__global__ void kernel(double *matrix, int row, int column, int n, int m) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;

	for (i = idy + column + 1; i < m; i += offsety)
		for (j = idx + row + 1; j < n; j += offsetx) 
			matrix[i * n + j] -= (matrix[column * n + j] / matrix[column * n + row]) * matrix[i * n + row];
}

int main() {
	std::ios_base::sync_with_stdio(false);

	int n, m;
	int i, j, column, row = 0, rank = 0;
	double max_element_row, eps = 0.0000001;
	double *matrix, *dev_matrix;
	comparator comp;

	std::cin >> n >> m;
	matrix = (double*)malloc(n * m * sizeof(double));

	for (i = 0; i < n; ++i)
		for (j = 0; j < m; ++j)
			std::cin >> matrix[j * n + i];

	CSC(cudaMalloc(&dev_matrix, n * m * sizeof(double)));
	CSC(cudaMemcpy(dev_matrix, matrix, n * m * sizeof(double), cudaMemcpyHostToDevice));
	free(matrix);

	thrust::device_ptr<double> matrix_p = thrust::device_pointer_cast(dev_matrix);
	for (column = 0; column < m && row < n; ++column) {
		thrust::device_ptr<double> max_p = thrust::max_element(matrix_p + n * column + row, matrix_p + n * (column + 1), comp);

		if (std::fabs(*max_p) <= eps)
			continue;

		max_element_row = max_p - (matrix_p + n * column);
		if (row != max_element_row) {
			kernel_swap_rows <<<256, 256>>> (dev_matrix, row, max_element_row, n, m);
			CSC(cudaGetLastError());
		}
		kernel <<<dim3(8, 8), dim3(32, 32)>>> (dev_matrix, row, column, n, m);
		CSC(cudaGetLastError());

		++rank;
		++row;
	}
	CSC(cudaFree(dev_matrix));
	std::cout << rank << "\n";

	return 0;
}