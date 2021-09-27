#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void ssaa(uchar4* res, int w, int h, int pw, int ph) {
	int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	int id_y = blockDim.y * blockIdx.y + threadIdx.y;
	int offset_x = blockDim.x * gridDim.x;
	int offset_y = blockDim.y * gridDim.y;
	
	uchar4 p;
	int cnt = pw * ph;
	for (int x = id_x; x < w; x += offset_x) {
		for (int y = id_y; y < h; y += offset_y) {
			uint3 sum = make_uint3(0, 0, 0);
			for (int i = 0; i < pw; ++i) {
				for (int j = 0; j < ph; ++j) {
					p = tex2D(tex, x * pw + i, y * ph + j);
					sum.x += p.x;
					sum.y += p.y;
					sum.z += p.z;
				}
			}
			res[y * w + x] = make_uchar4(sum.x / cnt, sum.y / cnt, sum.z / cnt, 0);
		}
	}
}

int main() {
	std::string in, out;
	int wn, hn, w, h;
	std::cin >> in >> out >> wn >> hn;
	
	FILE* fp = fopen(in.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = new uchar4[w * h];
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));
	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	delete[] data;

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.channelDesc = ch;
	tex.normalized = false;

	CSC(cudaBindTextureToArray(tex, arr, ch));
	uchar4* dev_res;
	CSC(cudaMalloc(&dev_res, sizeof(uchar4) * wn * hn));

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

    ssaa<<<dim3(16, 16), dim3(16, 16)>>> (dev_res, wn, hn, w / wn, h / hn);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
	printf("time = %f\n", t);

	uchar4* data_res = new uchar4[wn * hn];
	CSC(cudaMemcpy(data_res, dev_res, sizeof(uchar4) * wn * hn, cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_res));

	fp = fopen(out.c_str(), "wb");
	fwrite(&wn, sizeof(int), 1, fp);
	fwrite(&hn, sizeof(int), 1, fp);
	fwrite(data_res, sizeof(uchar4), wn * hn, fp);
	fclose(fp);

	delete[] data_res;
	return 0;
}