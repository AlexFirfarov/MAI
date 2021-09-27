#include <iostream>

__global__ void diff(double* v_1, double* v_2, double* res, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += offset)
		res[i] = v_1[i] - v_2[i];
}

int main() {
	int n = 0;
	std::cin >> n;

	double* v_1 = new double[n];
	double* v_2 = new double[n];
	double* res = new double[n];

	for (int i = 0; i < n; ++i)
		std::cin >> v_1[i];
	for (int i = 0; i < n; ++i)
		std::cin >> v_2[i];

	double *dev_1, *dev_2, *dev_res;
	cudaMalloc(&dev_1, sizeof(double) * n);
	cudaMalloc(&dev_2, sizeof(double) * n);
	cudaMalloc(&dev_res, sizeof(double) * n);

	cudaMemcpy(dev_1, v_1, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_2, v_2, sizeof(double) * n, cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	diff<<<256, 256>>>(dev_1, dev_2, dev_res, n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float t;
	cudaEventElapsedTime(&t, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("time = %f\n", t);

	cudaMemcpy(res, dev_res, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_1);
	cudaFree(dev_2);
	cudaFree(dev_res);

	delete[] v_1;
	delete[] v_2;

	std::cout.precision(10);
	std::cout.setf(std::ios::scientific);
	for (int i = 0; i < n; ++i)
		std::cout << res[i] << ' ';
	std::cout << '\n';
	delete[] res;
}
