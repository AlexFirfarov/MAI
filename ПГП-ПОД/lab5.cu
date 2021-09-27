#include <iostream>
#include <algorithm>
#include <cstdio>

using uint = unsigned int;
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _i(_index) ((_index) + ((_index) >> 5))

__global__ void kernel_digits(uint *data, int *radix_data, int n, int k) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int i;

	for (i = idx; i < n; i += offsetx)
		radix_data[i] = (data[i] >> k) & 1;
}

__global__ void kernel_radix_sort(uint *data, int *radix_data, uint *res, int n, int k) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int num_of_zeros = n - (radix_data[n - 1] + ((data[n - 1] >> k) & 1));
	int i, index1, index2, digit;

	for (i = idx; i < n; i += offsetx) {
		digit = (data[i] >> k) & 1;
		index1 = (radix_data[i] + num_of_zeros) * digit;
		index2 = (i - radix_data[i]) * (digit ^ 1);
		res[index1 + index2] = data[i];
	}
}

__global__ void kernel_scan(int *data, int *shift, int n) {
	__shared__ int s_data[BLOCK_SIZE + 8];
	int idx = threadIdx.x;
	int global_offset = (gridDim.x * blockIdx.y + blockIdx.x) * BLOCK_SIZE;
	int offset, s, index, tmp;

	s_data[_i(idx)] = ((global_offset + idx) < n) ? data[global_offset + idx] : 0;
	s_data[_i(idx + BLOCK_SIZE / 2)] = ((global_offset + idx + BLOCK_SIZE / 2) < n) ? data[global_offset + idx + BLOCK_SIZE / 2] : 0;

	for (s = 1; s <= BLOCK_SIZE / 2; s <<= 1) {
		__syncthreads();
		offset = s - 1;
		index = 2 * s * idx;
		if (index < BLOCK_SIZE)
			s_data[_i(offset + index + s)] += s_data[_i(offset + index)];
	}
	if (idx == 0) {
		shift[gridDim.x * blockIdx.y + blockIdx.x] = s_data[_i(BLOCK_SIZE - 1)];
		s_data[_i(BLOCK_SIZE - 1)] = 0;
	}
	for (s = BLOCK_SIZE / 2; s >= 1; s >>= 1) {
		__syncthreads();
		offset = s - 1;
		index = 2 * s * idx;
		if (index < BLOCK_SIZE) {
			tmp = s_data[_i(offset + index + s)];
			s_data[_i(offset + index + s)] += s_data[_i(offset + index)];
			s_data[_i(offset + index)] = tmp;
		}
	}
	__syncthreads();

	if ((global_offset + idx) < n)
		data[global_offset + idx] = s_data[_i(idx)];
	if ((global_offset + idx + BLOCK_SIZE / 2) < n)
		data[global_offset + idx + BLOCK_SIZE / 2] = s_data[_i(idx + BLOCK_SIZE / 2)];
}

__global__ void kernel_shift(int *data, int *shift, int n) {
	int idx = threadIdx.x;
	int offset = gridDim.x * blockIdx.y + blockIdx.x;
	int diff = shift[offset];
	offset *= BLOCK_SIZE;
	if (offset + idx < n)
		data[offset + idx] += diff;
	if (offset + idx + BLOCK_SIZE / 2 < n)
		data[offset + idx + BLOCK_SIZE / 2] += diff;
}

void scan(int *dev_data, int n) {
	int num_of_blocks = (n - 1) / BLOCK_SIZE + 1;
	dim3 blocks(std::min(num_of_blocks, MAX_BLOCKS), (num_of_blocks - 1) / MAX_BLOCKS + 1);
	dim3 threads(BLOCK_SIZE / 2);
	int *dev_shift;
	CSC(cudaMalloc(&dev_shift, num_of_blocks * sizeof(int)));

	kernel_scan <<<blocks, threads>>> (dev_data, dev_shift, n);
	CSC(cudaGetLastError());
	if (num_of_blocks == 1) {
		CSC(cudaFree(dev_shift));
		return;
	}
	scan(dev_shift, num_of_blocks);
	kernel_shift << <blocks, threads >> > (dev_data, dev_shift, n);
	CSC(cudaGetLastError());
	CSC(cudaFree(dev_shift));
}

int main() {
	int n, i;
	int *radix_data;
	uint *data, *dev_data, *dev_res, *tmp;

	fread(&n, sizeof(int), 1, stdin);

	data = (uint*)malloc(n * sizeof(uint));
	fread(data, sizeof(uint), n, stdin);

	CSC(cudaMalloc(&radix_data, n * sizeof(int)));
	CSC(cudaMalloc(&dev_res, n * sizeof(uint)));
	CSC(cudaMalloc(&dev_data, n * sizeof(uint)));
	CSC(cudaMemcpy(dev_data, data, n * sizeof(uint), cudaMemcpyHostToDevice));

	for (i = 0; i < 32; ++i) {
		kernel_digits <<<256, 256>>> (dev_data, radix_data, n, i);
		CSC(cudaGetLastError());
		scan(radix_data, n);
		kernel_radix_sort<<<256, 256>>> (dev_data, radix_data, dev_res, n, i);
		CSC(cudaGetLastError());

		tmp = dev_res;
		dev_res = dev_data;
		dev_data = tmp;
	}
	CSC(cudaMemcpy(data, dev_data, n * sizeof(uint), cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_res));
	CSC(cudaFree(radix_data));

	fwrite(data, sizeof(uint), n, stdout);

	free(data);

	return 0;
}