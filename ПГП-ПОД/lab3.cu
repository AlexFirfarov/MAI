#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__constant__ double dev_avg[32 * 3];
__constant__ double dev_det[32];
__constant__ double dev_ncov[32 * 9];

__device__ int classification(uchar4 p, int nc) {
	double d_i[32] = {};
	double t_1[3] = {};

	for (int i = 0; i < nc; ++i) {
		double t_2[3] = {};
		int offset = i * 9;

		t_1[0] = p.x - dev_avg[i * 3];
		t_1[1] = p.y - dev_avg[i * 3 + 1];
		t_1[2] = p.z - dev_avg[i * 3 + 2];

		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				t_2[j] += t_1[k] * dev_ncov[offset + k * 3 + j];
			}
			d_i[i] -= t_2[j] * t_1[j];
		}
		d_i[i] -= std::log(std::abs(dev_det[i]));
	}
	double d_max = d_i[0];
	int cl = 0;
	for (int i = 1; i < nc; ++i) {
		if (d_i[i] > d_max) {
			d_max = d_i[i];
			cl = i;
		}
	}
	return cl;
}

__global__ void MLE(uchar4* data, int w, int h, int nc) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for (int i = idx; i < w * h; i += offset) {
		data[i].w = classification(data[i], nc);
	}
}

int main() {
	std::string in, out;
	int nc, w, h;
	std::cin >> in >> out >> nc;
	std::vector<std::vector<uint2>> cl;
	for (int i = 0; i < nc; ++i) {
		int np;
		std::cin >> np;
		cl.push_back(std::vector<uint2>(np));
		for (int j = 0; j < np; ++j) {
			std::cin >> cl[i][j].x >> cl[i][j].y;
		}
	}

	FILE* fp = fopen(in.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = new uchar4[w * h];
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	double* avg = new double[nc * 3];
	std::fill_n(avg, nc * 3, 0.0);
	for (int i = 0; i < nc; ++i) {
		for (int j = 0; j < cl[i].size(); ++j) {
			uchar4 p = data[cl[i][j].y * w + cl[i][j].x];
			avg[i * 3] += p.x;
			avg[i * 3 + 1] += p.y;
			avg[i * 3 + 2] += p.z;
		}
		for (int k = 0; k < 3; ++k) {
			avg[i * 3 + k] /= cl[i].size();
		}
	}

	double* cov = new double[nc * 9];
	std::fill_n(cov, nc * 9, 0.0);
	for (int i = 0; i < nc; ++i) {
		int offset = i * 9;
		for (int j = 0; j < cl[i].size(); ++j) {
			uchar4 p = data[cl[i][j].y * w + cl[i][j].x];
			double v[3];
			v[0] = p.x - avg[i * 3];
			v[1] = p.y - avg[i * 3 + 1];
			v[2] = p.z - avg[i * 3 + 2];
			for (int n = 0; n < 3; ++n) {
				for (int m = 0; m < 3; ++m) {
					cov[offset + n * 3 + m] += v[n] * v[m];
				}
			}
		}
		for (int n_ = 0; n_ < 3; ++n_) {
			for (int m_ = 0; m_ < 3; ++m_) {
				cov[offset + n_ * 3 + m_] /= cl[i].size() - 1;
			}
		}
	}

	double* det = new double[nc];
	for (int i = 0; i < nc; ++i) {
		int offset = i * 9;
		det[i] = cov[offset] * (cov[offset + 4] * cov[offset + 8] - cov[offset + 7] * cov[offset + 5]) -
			cov[offset + 3] * (cov[offset + 1] * cov[offset + 8] - cov[offset + 7] * cov[offset + 2]) +
			cov[offset + 6] * (cov[offset + 1] * cov[offset + 5] - cov[offset + 4] * cov[offset + 2]);
	}

	//for (int i = 0; i < nc; ++i) std::cout << det[i] << " ";

	double* ncov = new double[nc * 9];
	for (int i = 0; i < nc; ++i) {
		int offset = i * 9;

		ncov[offset] = (cov[offset + 4] * cov[offset + 8] - cov[offset + 7] * cov[offset + 5]) / det[i];
		ncov[offset + 1] = -(cov[offset + 3] * cov[offset + 8] - cov[offset + 6] * cov[offset + 5]) / det[i];
		ncov[offset + 2] = (cov[offset + 3] * cov[offset + 7] - cov[offset + 6] * cov[offset + 4]) / det[i];
		ncov[offset + 3] = -(cov[offset + 1] * cov[offset + 8] - cov[offset + 7] * cov[offset + 2]) / det[i];
		ncov[offset + 4] = (cov[offset] * cov[offset + 8] - cov[offset + 6] * cov[offset + 2]) / det[i];
		ncov[offset + 5] = -(cov[offset] * cov[offset + 7] - cov[offset + 6] * cov[offset + 1]) / det[i];
		ncov[offset + 6] = (cov[offset + 1] * cov[offset + 5] - cov[offset + 4] * cov[offset + 2]) / det[i];
		ncov[offset + 7] = -(cov[offset] * cov[offset + 5] - cov[offset + 3] * cov[offset + 2]) / det[i];
		ncov[offset + 8] = (cov[offset] * cov[offset + 4] - cov[offset + 3] * cov[offset + 1]) / det[i];

		for (int m = 0; m < 3; ++m) {
			for (int n = m; n < 3; ++n) {
				std::swap(ncov[offset + m * 3 + n], ncov[offset + n * 3 + m]);
			}
		}
	}
	delete[] cov;

	uchar4* dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(dev_avg, avg, sizeof(double) * nc * 3));
	CSC(cudaMemcpyToSymbol(dev_det, det, sizeof(double) * nc));
	CSC(cudaMemcpyToSymbol(dev_ncov, ncov, sizeof(double) * nc * 9));

	delete[] avg;
	delete[] det;
	delete[] ncov;

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));
	MLE <<<16, 256 >>> (dev_data, w, h, nc);
	CSC(cudaGetLastError());
	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
	printf("time = %f\n", t);

	CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));

	fp = fopen(out.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	delete[] data;
	return 0;
}
