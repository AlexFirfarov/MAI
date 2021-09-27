#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mpi.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _i(i, j, k) ((((i) + 1) * (ny + 2) + ((j) + 1)) * (nz + 2) + ((k) + 1))

#define _ib(i, j, k) (((i) * nby + (j)) * nbz + (k))
#define _ibx(id) ((((id) / nbz) / nby))
#define _iby(id) ((((id) / nbz) % nby))
#define _ibz(id) (((id) % nbz))

__global__ void kernel_copy_to_buff_yz(double* data, double* buff, int nx, int ny, int nz, int x_c) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int j, k;

	for (j = idx; j < ny; j += offsetx)
		for (k = idy; k < nz; k += offsety)
			buff[j * nz + k] = data[_i(x_c, j, k)];
}

__global__ void kernel_copy_to_buff_xz(double* data, double* buff, int nx, int ny, int nz, int y_c) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int i, k;

	for (i = idx; i < nx; i += offsetx)
		for (k = idy; k < nz; k += offsety)
			buff[i * nz + k] = data[_i(i, y_c, k)];
}

__global__ void kernel_copy_to_buff_xy(double* data, double* buff, int nx, int ny, int nz, int z_c) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int i, j;

	for (i = idx; i < nx; i += offsetx)
		for (j = idy; j < ny; j += offsety)
			buff[i * ny + j] = data[_i(i, j, z_c)];
}

__global__ void kernel_copy_from_buff_yz(double* data, double* buff, int nx, int ny, int nz, int x_c, double bc) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int j, k;

	if (buff) {
		for (j = idx; j < ny; j += offsetx)
			for (k = idy; k < nz; k += offsety)
				data[_i(x_c, j, k)] = buff[j * nz + k];
	}
	else {
		for (j = idx; j < ny; j += offsetx)
			for (k = idy; k < nz; k += offsety)
				data[_i(x_c, j, k)] = bc;
	}
}

__global__ void kernel_copy_from_buff_xz(double* data, double* buff, int nx, int ny, int nz, int y_c, double bc) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int i, k;

	if (buff) {
		for (i = idx; i < nx; i += offsetx)
			for (k = idy; k < nz; k += offsety)
				data[_i(i, y_c, k)] = buff[i * nz + k];
	}
	else {
		for (i = idx; i < nx; i += offsetx)
			for (k = idy; k < nz; k += offsety)
				data[_i(i, y_c, k)] = bc;
	}
}

__global__ void kernel_copy_from_buff_xy(double* data, double* buff, int nx, int ny, int nz, int z_c, double bc) {
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.x * gridDim.x;
	int offsetx = blockDim.y * gridDim.y;
	int i, j;

	if (buff) {
		for (i = idx; i < nx; i += offsetx)
			for (j = idy; j < ny; j += offsety)
				data[_i(i, j, z_c)] = buff[i * ny + j];
	}
	else {
		for (i = idx; i < nx; i += offsetx)
			for (j = idy; j < ny; j += offsety)
				data[_i(i, j, z_c)] = bc;
	}
}

__global__ void kernel(double* data, double* next, double* errors, int nx, int ny, int nz, double hx, double hy, double hz) {
	int idz = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetz = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int offsetx = blockDim.z * gridDim.z;
	for (int i = idx; i < nx; i += offsetx)
		for (int j = idy; j < ny; j += offsety)
			for (int k = idz; k < nz; k += offsetz) {
				next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
					(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
					(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
					(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
				errors[i * ny * nz + j * nz + k] = std::fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
			}
}

int main(int argc, char* argv[]) {
	int ib, jb, kb;
	int i, j, k;
	int id_proc, nbx, nby, nbz, nx, ny, nz;
	double max_err, err;
	double hx, hy, hz;
	double eps, lx, ly, lz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u_0;
	double *data, *dev_data, *next, *dev_next, *buff, *dev_buff, *dev_errors, *temp, *recvbuf_err;
	char *file_buff;
	char file_name[100];

	int device_count;
	cudaGetDeviceCount(&device_count);

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
	MPI_Barrier(MPI_COMM_WORLD);

	cudaSetDevice(id_proc % device_count);

	if (id_proc == 0) {
		std::cin >> nbx >> nby >> nbz;
		std::cin >> nx >> ny >> nz;
		std::cin >> file_name;
		std::cin >> eps;
		std::cin >> lx >> ly >> lz;
		std::cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back;
		std::cin >> u_0;
	}

	MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&file_name, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	ib = _ibx(id_proc);
	jb = _iby(id_proc);
	kb = _ibz(id_proc);

	hx = lx / (nx * nbx);
	hy = ly / (ny * nby);
	hz = lz / (nz * nbz);

	int n_max = std::max(nx, std::max(ny, nz));

	data = (double*)malloc((nx + 2) * (ny + 2) * (nz + 2) * sizeof(double));
	next = (double*)malloc((nx + 2) * (ny + 2) * (nz + 2) * sizeof(double));
	buff = (double*)malloc(n_max * n_max * sizeof(double));
	recvbuf_err = (double*)malloc(nbx * nby * nbz * sizeof(double));

	CSC(cudaMalloc(&dev_data, (nx + 2) * (ny + 2) * (nz + 2) * sizeof(double)));
	CSC(cudaMalloc(&dev_next, (nx + 2) * (ny + 2) * (nz + 2) * sizeof(double)));
	CSC(cudaMalloc(&dev_buff, n_max * n_max * sizeof(double)));
	CSC(cudaMalloc(&dev_errors, nx * ny * nz * sizeof(double)));

	int buffer_size = 6 * (n_max * n_max * sizeof(double) + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	for (i = 0; i < nx; ++i)
		for (j = 0; j < ny; ++j)
			for (k = 0; k < nz; ++k)
				data[_i(i, j, k)] = u_0;

	CSC(cudaMemcpy(dev_data, data, (nx + 2) * (ny + 2) * (nz + 2) * sizeof(double), cudaMemcpyHostToDevice));

	dim3 blocks(8, 8);
	dim3 threads(32, 4);

	for (;;) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {
			kernel_copy_to_buff_yz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, nx - 1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id_proc, MPI_COMM_WORLD);
		}
		if (ib > 0) {
			kernel_copy_to_buff_yz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id_proc, MPI_COMM_WORLD);
		}
		if (jb + 1 < nby) {
			kernel_copy_to_buff_xz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, ny - 1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, nx * nz * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id_proc, MPI_COMM_WORLD);
		}
		if (jb > 0) {
			kernel_copy_to_buff_xz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, nx * nz * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id_proc, MPI_COMM_WORLD);
		}
		if (kb + 1 < nbz) {
			kernel_copy_to_buff_xy << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, nz - 1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, nx * ny * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), id_proc, MPI_COMM_WORLD);
		}
		if (kb > 0) {
			kernel_copy_to_buff_xy << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, nx * ny * sizeof(double), cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), id_proc, MPI_COMM_WORLD);
		}

		if (ib + 1 < nbx) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, ny * nz * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_yz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, nx, 0.0);

		}
		else {
			kernel_copy_from_buff_yz << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, nx, bc_right);
		}
		CSC(cudaGetLastError());

		if (ib > 0) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, ny * nz * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_yz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, -1, 0.0);
		}
		else {
			kernel_copy_from_buff_yz << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, -1, bc_left);
		}
		CSC(cudaGetLastError());

		if (jb + 1 < nby) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, nx * nz * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_xz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, ny, 0.0);
		}
		else {
			kernel_copy_from_buff_xz << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, ny, bc_back);
		}
		CSC(cudaGetLastError());

		if (jb > 0) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, nx * nz * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_xz << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, -1, 0.0);
		}
		else {
			kernel_copy_from_buff_xz << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, -1, bc_front);
		}
		CSC(cudaGetLastError());

		if (kb + 1 < nbz) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, nx * ny * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_xy << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, nz, 0.0);
		}
		else {
			kernel_copy_from_buff_xy << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, nz, bc_up);
		}
		CSC(cudaGetLastError());

		if (kb > 0) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, nx * ny * sizeof(double), cudaMemcpyHostToDevice));
			kernel_copy_from_buff_xy << <blocks, threads >> > (dev_data, dev_buff, nx, ny, nz, -1, 0.0);
		}
		else {
			kernel_copy_from_buff_xy << <blocks, threads >> > (dev_data, NULL, nx, ny, nz, -1, bc_down);
		}
		CSC(cudaGetLastError());

		MPI_Barrier(MPI_COMM_WORLD);

		kernel << <dim3(4, 4, 4), dim3(32, 4, 4) >> > (dev_data, dev_next, dev_errors, nx, ny, nz, hx, hy, hz);
		CSC(cudaGetLastError());

		err = 0.0;
		thrust::device_ptr<double> errors_p = thrust::device_pointer_cast(dev_errors);
		thrust::device_ptr<double> err_p = thrust::max_element(errors_p, errors_p + nx * ny * nz);
		err = *err_p;

		temp = dev_next;
		dev_next = dev_data;
		dev_data = temp;

		MPI_Allgather(&err, 1, MPI_DOUBLE, recvbuf_err, 1, MPI_DOUBLE, MPI_COMM_WORLD);

		max_err = 0.0;
		for (i = 0; i < nbx * nby * nbz; ++i) {
			if (recvbuf_err[i] > max_err)
				max_err = recvbuf_err[i];
		}

		if (max_err < eps)
			break;
	}

	CSC(cudaMemcpy(data, dev_data, (nx + 2) * (ny + 2) * (nz + 2) * sizeof(double), cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_next));
	CSC(cudaFree(dev_buff));
	CSC(cudaFree(dev_errors));

	MPI_Barrier(MPI_COMM_WORLD);

	int n_size = 20;
	file_buff = (char*)malloc(nx * ny * nz * n_size * sizeof(char));
	memset(file_buff, ' ', nx * ny * nz * n_size * sizeof(char));

	for (k = 0; k < nz; ++k)
		for (j = 0; j < ny; ++j) {
			for (i = 0; i < nx; ++i)
				sprintf(file_buff + (k * ny * nx + j * nx + i) * n_size, "%e ", data[_i(i, j, k)]);
			if (ib + 1 == nbx) {
				file_buff[(k * ny * nx + j * nx + nx) * n_size - 1] = '\n';
				if (jb + 1 == nby && j + 1 == ny)
					file_buff[(k * ny * nx + j * nx + nx) * n_size - 2] = '\n';
			}
		}

	for (i = 0; i < nx * ny * nz * n_size; ++i)
		if (file_buff[i] == '\0')
			file_buff[i] = ' ';

	MPI_File fp;
	MPI_Datatype type1;
	MPI_Datatype type2;

	MPI_Type_create_hvector(ny, nx * n_size * sizeof(char), nx * n_size * nbx * sizeof(char), MPI_CHAR, &type1);
	MPI_Type_commit(&type1);

	MPI_Type_create_hvector(nz, 1, nby * nx * ny * n_size * nbx * sizeof(char), type1, &type2);
	MPI_Type_commit(&type2);

	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, (kb * nbx * nby * nz + jb * nbx) * (nx * ny * n_size * sizeof(char)) + ib * nx * n_size * sizeof(char),
		MPI_CHAR, type2, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, file_buff, nx * ny * nz * n_size * sizeof(char), MPI_CHAR, MPI_STATUS_IGNORE);

	MPI_File_close(&fp);
	MPI_Type_free(&type1);
	MPI_Type_free(&type2);

	MPI_Buffer_detach(buffer, &buffer_size);
	MPI_Finalize();

	free(buff);
	free(data);
	free(next);
	free(buffer);
	free(file_buff);
	free(recvbuf_err);

	return 0;
}
