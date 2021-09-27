#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>

#include "mpi.h"

#define _i(i, j, k) ((((i) + 1) * (ny + 2) + ((j) + 1)) * (nz + 2) + ((k) + 1))
#define _ix(id) ((((id) / (nz + 2)) / (ny + 2) - 1))
#define _iy(id) ((((id) / (nz + 2)) % (ny + 2) - 1))
#define _iz(id) (((id) % (nz + 2) - 1))

#define _ib(i, j, k) (((i) * nby + (j)) * nbz + (k))
#define _ibx(id) ((((id) / nbz) / nby))
#define _iby(id) ((((id) / nbz) % nby))
#define _ibz(id) (((id) % nbz))

int main(int argc, char* argv[]) {
	int ib, jb, kb;
	int i, j, k;
	int id_proc, nbx, nby, nbz, nx, ny, nz;
	std::string out;
	double max_err, err;
	double hx, hy, hz;
	double eps, lx, ly, lz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u_0;
	double *data, *temp, *next, *buff, *recvbuf_err;
	double sendbuf_err[1];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
	MPI_Barrier(MPI_COMM_WORLD);

	if (id_proc == 0) {
		std::cin >> nbx >> nby >> nbz;
		std::cin >> nx >> ny >> nz;
		std::cin >> out;
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

	int buffer_size = 6 * n_max * n_max * sizeof(double) + MPI_BSEND_OVERHEAD;
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	for (i = 0; i < nx; ++i)			
		for (j = 0; j < ny; ++j)
			for (k = 0; k < nz; ++k)
				data[_i(i, j, k)] = u_0;
			
	for (;;) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					buff[j * nz + k] = data[_i(nx - 1, j, k)];
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id_proc, MPI_COMM_WORLD);
		}
		if (ib > 0) {
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					buff[j * nz + k] = data[_i(0, j, k)];
			MPI_Bsend(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id_proc, MPI_COMM_WORLD);
		}
		if (jb + 1 < nby) {
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					buff[i * nz + k] = data[_i(i, ny - 1, k)];
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id_proc, MPI_COMM_WORLD);
		}
		if (jb > 0) {
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					buff[i * nz + k] = data[_i(i, 0, k)];
			MPI_Bsend(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id_proc, MPI_COMM_WORLD);
		}
		if (kb + 1 < nbz) {
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					buff[i * ny + j] = data[_i(i, j, nz - 1)];
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), id_proc, MPI_COMM_WORLD);
		}
		if (kb > 0) {
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					buff[i * ny + j] = data[_i(i, j, 0)];
			MPI_Bsend(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), id_proc, MPI_COMM_WORLD);
		}

		if (ib + 1 < nbx) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					data[_i(nx, j, k)] = buff[j * nz + k];
		} else {
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					data[_i(nx, j, k)] = bc_right;
		}

		if (ib > 0) {
			MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					data[_i(-1, j, k)] = buff[j * nz + k];
		} else {
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k)
					data[_i(-1, j, k)] = bc_left;
		}

		if (jb + 1 < nby) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					data[_i(i, ny, k)] = buff[i * nz + k];
		} else {
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					data[_i(i, ny, k)] = bc_back;
		}

		if (jb > 0) {
			MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					data[_i(i, -1, k)] = buff[i * nz + k];
		} else {
			for (i = 0; i < nx; ++i)
				for (k = 0; k < nz; ++k)
					data[_i(i, -1, k)] = bc_front;
		}

		if (kb + 1 < nbz) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					data[_i(i, j, nz)] = buff[i * ny + j];
		} else {
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					data[_i(i, j, nz)] = bc_up;
		}

		if (kb > 0) {
			MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					data[_i(i, j, -1)] = buff[i * ny + j];
		} else {
			for (i = 0; i < nx; ++i)
				for (j = 0; j < ny; ++j)
					data[_i(i, j, -1)] = bc_down;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		max_err = 0.0;
		for (i = 0; i < nx; ++i)
			for (j = 0; j < ny; ++j)
				for (k = 0; k < nz; ++k) {
					next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
											   (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
						                       (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
						                       (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
					err = std::abs(next[_i(i, j, k)] - data[_i(i, j, k)]);
					if (err > max_err)
						max_err = err;
				}

		temp = next;
		next = data;
		data = temp;

		sendbuf_err[0] = max_err;
		MPI_Allgather(sendbuf_err, 1, MPI_DOUBLE, recvbuf_err, 1, MPI_DOUBLE, MPI_COMM_WORLD);

		max_err = 0.0;
		for (i = 0; i < nbx * nby * nbz; ++i) {
			if (recvbuf_err[i] > max_err)
				max_err = recvbuf_err[i];
		}
		if (max_err < eps)
			break;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (id_proc != 0) {
		for (k = 0; k < nz; ++k) {
			for (j = 0; j < ny; ++j) {
				for (i = 0; i < nx; ++i)
					buff[i] = data[_i(i, j, k)];
				MPI_Send(buff, nx, MPI_DOUBLE, 0, id_proc, MPI_COMM_WORLD);
			}
		}
	} else {
		std::ofstream file(out);
		file.precision(6);
		file.setf(std::ios::scientific);
		for (kb = 0; kb < nbz; ++kb)
			for (k = 0; k < nz; ++k)
				for (jb = 0; jb < nby; ++jb)
					for (j = 0; j < ny; ++j)
						for (ib = 0; ib < nbx; ++ib) {
							if (_ib(ib, jb, kb) == 0)
								for (i = 0; i < nx; ++i)
									buff[i] = data[_i(i, j, k)];
							else
								MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							for (i = 0; i < nx; ++i)
								file << buff[i] << " ";
							if (ib + 1 == nbx) {
								file << "\n";
								if (j + 1 == ny) 
									file << "\n";
							} else {
								file << " ";
							}
						}
		file.close();
	}

	MPI_Buffer_detach(buffer, &buffer_size);
	MPI_Finalize();

	free(buff);
	free(data);
	free(next);
	free(buffer);
	free(recvbuf_err);

	return 0;
}