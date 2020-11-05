#pragma once

template <typename T>
T* cudaCopy(T* in, unsigned int s = sizeof(T)) {
	T* t;
	cudaMalloc((void**)&t, s);
	cudaMemcpy(t, in, s, cudaMemcpyHostToDevice);
	return t;
}

#include <iostream>
#include <vector>
#include <list>

using namespace std;

namespace SLA {

	template <typename T>
	class CuVec {
	public:
		T* d_a;
		list<T> a;
		vector<T> va;
		unsigned int capacity, size, *d_i;
		list<unsigned int> indexes;
		vector<unsigned int> vindexes;

		CuVec() {
			d_a = 0;
			size = 0;
		}
		void reserve(unsigned int s) {
			size = 0;
			capacity = s;
			a.clear();
			indexes.clear();
		}
		CuVec* upload() {
			va.resize(size);
			vindexes.resize(size);
			int ind = 0;
			list<unsigned int>::iterator i;
			typename list<T>::iterator j;
			for (i = indexes.begin(), j = a.begin(); i != indexes.end() && j != a.end(); ++i, ++j, ++ind) {
				va[ind] = *j;
				vindexes[ind] = *i;
			}
			cudaMalloc((void**)&d_a, sizeof(T)*size);
			cudaMalloc((void**)&d_i, sizeof(unsigned int)*size);
			cudaMemcpy(d_i, vindexes.data(), sizeof(unsigned int)*size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_a, va.data(), sizeof(T)*size, cudaMemcpyHostToDevice);
			return cudaCopy(this);
		}
		void download() {
			cudaMemcpy(vindexes.data(), d_i, sizeof(unsigned int)*size, cudaMemcpyDeviceToHost);
			cudaMemcpy(va.data(), d_a, sizeof(T)*size, cudaMemcpyDeviceToHost);
			free();
		}
		void set(const unsigned int& index, const T& v) {
			if (index > capacity) {
				exit(1);
			}
			list<unsigned int>::iterator i;
			typename list<T>::iterator j;
			for (i = indexes.begin(), j = a.begin(); j != a.end() && i != indexes.end() && *i < index; ++i, ++j);
			if (a.size() && *i == index) {
				if(v != 0)
					*j = v;
				else {
					a.remove(j);
					indexes.remove(i);
				}
			}
			else {
				if (v != 0) {
					a.insert(j, v);
					indexes.insert(i, index);
					++size;
				}
			}
		}
		void setIncremental(const unsigned int& index, const T& v) {//special case, use only with empty arrays when filling with a forall style
			if (index > capacity)
				exit(1);
			a.push_back(v);
			indexes.push_back(index);
			++size;
		}
		void free(){
			cudaFree(d_a);
			cudaFree(d_i);
			d_a = 0;
			d_i = 0;
		}
		void print() {
			printf("Vector ");
			for (typename list<T>::iterator i = a.begin(); i != a.end(); ++i) {
				printf("%f ", *i);
			}
			printf("\n");
		}
		__host__ T& get(unsigned int ind) {
			list<T>::iterator i;
			list<unsigned int>::iterator j;
			for (i = a.begin(), j = indexes.begin(); i != a.end() && j != indexes.end(); ++i, ++j) {
				if (ind == *j)
					return *i;
			}
			return 0;
		}
		__device__ T& d_get(unsigned int ind) {
			int l = 0, c = size / 2, h = size - 1;
			if (ind > d_i[h] || ind < d_i[l]) {
				return 0;
			}
			for (; l != h;) {
				if (d_i[c] == ind)
					return d_a[c];
				if (ind > d_i[c]) {
					l = c;
				}
				else h = c;
				c = (h + l) / 2;
			}
			return 0;
		}
		__host__ T& operator[](const int& i, const T& val) {
			set(i, val);
			return get(i);
		}
		__host__ const T& operator[](const int& i) {
			return get(i);
		}
		__device__ const T& operator[](const int& i) {
			return d_get(i);
		}
		__device__ void d_set(unsigned int index, const T& val) {
			T& v = get(index);
			if(v != 0)
				v = val;
		}
	};

	template <typename T>
	class CuMat {
	public:
		T* d_a;
		list<T> a;
		list<unsigned int> c;
		vector<unsigned int> r, vc;
		vector<T> va;
		unsigned int size, numR, numC, *d_r, *d_c;

		__device__ __host__ CuMat() {
			size = 0;
			d_a = 0;
			numR = 0;
			numC = 0;
			d_r = 0;
			d_c = 0;
		}
		void resize(unsigned int cols, unsigned int rows) {
			numR = rows;
			numC = cols;
			c.clear();
			r.resize(numR + 1);
			a.clear();
			size = 0;
		}
		void set(const unsigned int& col, const unsigned int& row, const T& v) {
			if (col >= numC || row >= numR) {
				exit(1);
			}
			int i = r[row];
			typename list<T>::iterator it = a.begin();
			list<unsigned int>::iterator ct = c.begin();
			for (int j = 0; j < i && it != a.end() && ct != c.end(); ++it, ++ct, ++j);
			a.insert(it, v);
			c.insert(ct, col);
			for (int j = row+1; j < numR+1; ++j) {
				++r[j];
			}
			++size;
		}
		__device__ __host__ bool get(unsigned int col, unsigned int row, T& out) {
			for (int i = r[row]; i < r[row + 1] && c[i] <= col; ++i) {
				if (c[i] == col) {
					out = a[i];
					return true;
				}
			}
			return false;
		}
		__device__ T& d_get(unsigned int col, unsigned int row) {
			//int l = d_r[row + 1] - d_r[row];
		}
		void print() {
			printf("V: ");
			for (typename list<T>::iterator i = a.begin(); i != a.end(); ++i) {
				printf("%f ", *i);
			}
			printf("\n");

			printf("C: ");
			for (list<unsigned int>::iterator i = c.begin(); i != c.end(); ++i) {
				printf("%d ", *i);
			}
			printf("\n");

			printf("R: ");
			for (int i = 0; i < numR+1; ++i) {
				printf("%d ", r[i]);
			}
			printf("\n");
		}
		CuMat* upload() {
			va.resize(size);
			vc.resize(size);

			typename list<T>::iterator i = a.begin();
			list<unsigned int>::iterator j = c.begin();
			for (int k = 0; i != a.end() && j != c.end(); ++i, ++j, ++k) {
				va[k] = *i;
				vc[k] = *j;
			}

			d_a = cudaCopy(va.data(), sizeof(T)*size);
			d_c = cudaCopy(vc.data(), sizeof(unsigned int)*size);
			d_r = cudaCopy(r.data(), sizeof(unsigned int)*size);
			return cudaCopy(this);
		}
		void download() {
			cudaMemcpy(va.data(), d_a, sizeof(T)*size, cudaMemcpyDeviceToHost);
			cudaMemcpy(vc.data(), d_c, sizeof(unsigned int)*size, cudaMemcpyDeviceToHost);
			cudaMemcpy(r.data(), d_r, sizeof(unsigned int)*(numR + 1), cudaMemcpyDeviceToHost);
			free();
		}
		void free() {
			cudaFree(d_a);
			cudaFree(d_c);
			cudaFree(d_r);
			d_a = 0;
			d_c = 0;
			d_r = 0;
		}
	};

	template <typename T>
	__global__ void matvec(CuMat<T>* m, CuVec<T>* v, T* out) {//sparse mat, sparse invec, dense outvec
		if (v->capacity != m->numC) {
			return;
		}
		unsigned int offset = blockDim.x*gridDim.x;
		for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < m->numR; i += offset) {
			out[i] = 0;
			for (int j = m->d_r[i], cc = 0; j < m->d_r[i + 1]; ++j) {
				while (v->d_i[cc] < m->d_c[j]) {
					++cc;
				}
				--cc;
				if (cc >= 0 && v->d_i[cc] == m->d_c[j]) {
					out[i] += m->d_a[j] * v->d_a[cc];
				}
			}
		}
	}
}