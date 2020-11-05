#include "TriVec.h"
using namespace std;
#include <iostream>
#include <float.h>

using namespace Cu;
template <typename T>
TriVec<T>::TriVec() {
	size = 0;
	a = nullptr;
	x = y = z = 0;
	invalid = Voxel<T>(true);
	invalidX = 0;
	invalidY = 0;
	invalidZ = 0;
	maxRes = 0;
	density = 0;
	numSolids = 0;
	res = nullptr;
	this->numPFluids = 0;
	this->pFluidIndexes = nullptr;

#ifndef __CUDA_ARCH__
	this->dist = new std::uniform_real_distribution<double>(0.0, 1.0);
	this->gen = new std::default_random_engine();
#else
	this->dist = nullptr;
	this->gen = nullptr;
#endif
}

template <typename T>
TriVec<T>::TriVec(unsigned long long x, unsigned long long y, unsigned long long z, double dx, double dt, double density, bool dev) : TriVec() {
	this->size = x*y*z;
	if(!dev){
		this->a = new Voxel<T>[size];
		this->res = new T[size];
		this->pFluidIndexes = new int[size];
	}
	else{
		this->a = nullptr;
		this->res = nullptr;
		this->pFluidIndexes = nullptr;
	}
	this->dx = dx;
	this->x = x;
	this->y = y;
	this->z = z;
	invalid = Voxel<T>(true);
	this->density = density;
	this->dt = dt;

	matrix.resize(size, size);
	pressure.reserve(size);
	aX.reserve(size);
	aY.reserve(size);
	aZ.reserve(size);
	aDiag.reserve(size);
	anX.reserve(size);
	anY.reserve(size);
	anZ.reserve(size);
	uX.reserve(size);
	uY.reserve(size);
	uZ.reserve(size);
}

template <typename T>
TriVec<T>::~TriVec() {
#ifndef __CUDA_ARCH__
	if (a != nullptr) {
		delete[] a;
	}
	if (res != nullptr) {
		delete[] res;
	}
#endif
}

template <typename T>
const TriVec<T>& TriVec<T>::operator=(const TriVec<T>& in) {
	size = in.size;
	a = in.a;

	x = in.x;
	y = in.y;
	z = in.z;
	numSolids = in.numSolids;
	dx = in.dx;
	res = in.res;

	this->numPFluids = in.numPFluids;
	pFluidIndexes = in.pFluidIndexes;

	/*for (int i = 0; i < 12; ++i) {
		this->particles[i] = in.particles[i];
	}*/

	return *this;
}

template <typename T>
const TriVec<T>& TriVec<T>::operator=(TriVec<T>&& in){	//move assignment op, useful for copying when you need to preserve x,y,z and dont actually need a copy, especially when instantiating member TriVecs
	size = in.size;
	a = in.a;
	in.a = nullptr;
	
	x = in.x;
	y = in.y;
	z = in.z;
	numSolids = in.numSolids;
	dx = in.dx;

	res = in.res;
	in.res = nullptr;

	this->numPFluids = in.numPFluids;
	pFluidIndexes = in.pFluidIndexes;
	in.pFluidIndexes = nullptr;

	return *this;
}

template <typename T>
Voxel<T>& TriVec<T>::get(int x, int y, int z){
	if(x >= 0 && x < this->x && y >= 0 && y < this->y && z >= 0 && z < this->z)
		return a[z*this->x*this->y + y*this->x + x];
	return invalid;
}

template <typename T>
__device__ T TriVec<T>::kWeight(Vec3<T> in){
	float tx = abs(in.x/dx), ty = abs(in.y/dx), tz = abs(in.z/dx);
	if(tx >= 0 && tx < 0.5)
		tx = .75-tx*tx;
	else if(tx >= 0.5 && tx < 1.5)
		tx = 0.5*(1.5-tx)*(1.5-tx);
	else tx = 0;

	if(ty >= 0 && ty < 0.5)
		ty = .75-ty*ty;
	else if(ty >= 0.5 && ty < 1.5)
		ty = 0.5*(1.5-ty)*(1.5-ty);
	else ty = 0;

	if(tz >= 0 && tz < 0.5)
		tz = .75-tz*tz;
	else if(tz >= 0.5 && tz < 1.5)
		tz = 0.5*(1.5-tz)*(1.5-tz);
	else tz = 0;

	return tx*ty*tz;
}

template <typename T>
void TriVec<T>::interpU() {
#ifdef __CUDA_ARCH__
	int o = gridDim.x*blockDim.x;
	for (int l = threadIdx.x + blockDim.x*blockIdx.x; l < size; l += o) {
		if (a[l].t != SOLID) {
			a[l].u.x = a[l].sumuk.x / (a[l].sumk.x + 0.0000000000000001);
			a[l].u.y = a[l].sumuk.y / (a[l].sumk.y + 0.0000000000000001);
			a[l].u.z = a[l].sumuk.z / (a[l].sumk.z + 0.0000000000000001);

			int k = l / (x*y), j = (l % (x*y)) / x, i = (l % (x*y)) % x;

			if (get(i + 1, j, k).t == SOLID) {
				a[l].u.x = 0;
			}
			if (get(i, j + 1, k).t == SOLID) {
				a[l].u.y = 0;
			}
			if (get(i, j, k + 1).t == SOLID) {
				a[l].u.z = 0;
			}
		}
	}
#else
//#pragma omp parallel for
	//for (int l = 0; l < size; ++l) {
	concurrency::parallel_for(size_t(0), size_t(size), [&](int l){
		a[l].sumuk.x /= (a[l].sumk.x + 0.0000000000000000001);
		a[l].sumuk.y /= (a[l].sumk.y + 0.0000000000000000001);
		a[l].sumuk.z /= (a[l].sumk.z + 0.0000000000000000001);
		a[l].u = a[l].sumuk;
		a[l].sumuk = Vec3<T>();
		a[l].sumk = Vec3<T>();
		int k = l / (x*y), j = (l % (x*y)) / x, i = (l % (x*y)) % x;
		if (get(i + 1, j, k).t == SOLID) {
			a[l].u.x = 0;
		}
		if (get(i, j + 1, k).t == SOLID) {
			a[l].u.y = 0;
		}
		if (get(i, j, k + 1).t == SOLID) {
			a[l].u.z = 0;
		}
		a[l].uOld = a[l].u;
	});
#endif
}

#define ALPHA 0.97											//defines amount of FLIP to use, 1 is all FLIP, 0 is no FLIP
template <typename T>
void TriVec<T>::interpUtoP(Particle<T>& in){		//interpolate surrounding grid velocities to particles
	Vec3<T> newU, oldU;
	int tx = in.p.x/dx, ty = in.p.y/dx, tz = in.p.z/dx;		//get xyz of particle's voxel
	if(in.p.y - ty*dx < dx/2){								//x component of velocity is stored on upper x bound and halfway point in y and z, need to adjust whether y and z are from y/z and y+1/z+1 or y-1/z-1 and y/z for trilin interp
		--ty;
	}
	if(in.p.z - tz*dx < dx/2){
		--tz;
	}
	
	newU.x = trilinterp((in.p.x - tx*dx) / dx, (in.p.y - (ty*dx + dx/2))/dx, (in.p.z - (tz*dx + dx/2))/dx, get(tx-1, ty, tz).u.x, get(tx, ty, tz).u.x, get(tx-1, ty+1, tz).u.x, get(tx, ty+1, tz).u.x, get(tx-1, ty, tz+1).u.x, get(tx, ty, tz+1).u.x, get(tx-1, ty+1, tz+1).u.x, get(tx, ty+1, tz+1).u.x);
	oldU.x = trilinterp((in.p.x - tx*dx) / dx, (in.p.y - (ty*dx + dx/2))/dx, (in.p.z - (tz*dx + dx/2))/dx, get(tx-1, ty, tz).uOld.x, get(tx, ty, tz).uOld.x, get(tx-1, ty+1, tz).uOld.x, get(tx, ty+1, tz).uOld.x, get(tx-1, ty, tz+1).uOld.x, get(tx, ty, tz+1).uOld.x, get(tx-1, ty+1, tz+1).uOld.x, get(tx, ty+1, tz+1).uOld.x);
	
	tx = in.p.x/dx, ty = in.p.y/dx, tz = in.p.z/dx;
	if(in.p.x - tx*dx < dx/2){
		--tx;
	}
	if(in.p.z - tz*dx < dx/2){
		--tz;
	}
	newU.y = trilinterp((in.p.x - (tx*dx + dx/2))/dx, (in.p.y - ty*dx)/dx, (in.p.z - (tz*dx + dx/2))/dx, get(tx, ty - 1, tz).u.y, get(tx + 1, ty - 1, tz).u.y, get(tx, ty, tz).u.y, get(tx + 1, ty, tz).u.y, get(tx, ty - 1, tz + 1).u.y, get(tx + 1, ty - 1, tz + 1).u.y, get(tx, ty, tz + 1).u.y, get(tx + 1, ty, tz + 1).u.y);
	oldU.y = trilinterp((in.p.x - (tx*dx + dx/2))/dx, (in.p.y - ty*dx)/dx, (in.p.z - (tz*dx + dx/2))/dx, get(tx, ty - 1, tz).uOld.y, get(tx + 1, ty - 1, tz).uOld.y, get(tx, ty, tz).uOld.y, get(tx + 1, ty, tz).uOld.y, get(tx, ty - 1, tz + 1).uOld.y, get(tx + 1, ty - 1, tz + 1).uOld.y, get(tx, ty, tz + 1).uOld.y, get(tx + 1, ty, tz + 1).uOld.y);

	tx = in.p.x/dx, ty = in.p.y/dx, tz = in.p.z/dx;
	if(in.p.x - tx*dx < dx/2){
		--tx;
	}
	if(in.p.y - ty*dx < dx/2){
		--ty;
	}
	newU.z = trilinterp((in.p.x - (tx*dx + dx/2))/dx, (in.p.y - (ty*dx + dx/2))/dx, (in.p.z - tz*dx)/dx, get(tx, ty, tz - 1).u.z, get(tx + 1, ty, tz - 1).u.z, get(tx, ty + 1, tz - 1).u.z, get(tx + 1, ty + 1, tz - 1).u.z, get(tx, ty, tz).u.z, get(tx + 1, ty, tz).u.z, get(tx, ty + 1, tz).u.z, get(tx + 1, ty + 1, tz).u.z);
	oldU.z = trilinterp((in.p.x - (tx*dx + dx/2))/dx, (in.p.y - (ty*dx + dx/2))/dx, (in.p.z - tz*dx)/dx, get(tx, ty, tz - 1).uOld.z, get(tx + 1, ty, tz - 1).uOld.z, get(tx, ty + 1, tz - 1).uOld.z, get(tx + 1, ty + 1, tz - 1).uOld.z, get(tx, ty, tz).uOld.z, get(tx + 1, ty, tz).uOld.z, get(tx, ty + 1, tz).uOld.z, get(tx + 1, ty + 1, tz).uOld.z);
	
	in.v = (1-ALPHA)*newU + ALPHA*(in.v + (newU - oldU));
}

template <typename T>
__device__ T TriVec<T>::divergenceU(int x, int y, int z){		//calc right hand side, modified to only allow for non-solid surfaces
	Vec3<T> t = get(x, y, z).u;
	double scale = -1 / dx, div = 0;
	if(get(x-1, y, z).t != SOLID)
		div -= get(x - 1, y, z).u.x;
	if(get(x+1, y, z).t != SOLID)
		div += t.x;
	if(get(x, y-1, z).t != SOLID)
		div -= get(x, y - 1, z).u.y;
	if(get(x, y+1, z).t != SOLID)
		div += t.y;
	if(get(x, y, z-1).t != SOLID)
		div -= get(x, y, z - 1).u.z;
	if(get(x, y, z+1).t != SOLID)
		div += t.z;

	div *= scale;
	return div;
}

template <typename T>
__global__ void cuDivergenceU(CuVec<T>* Ux, CuVec<T>* Uy, CuVec<T>* Uz, CuVec<T>* divU, int x, int y, int z) {
	unsigned int offset = gridDim.x*blockDim.x, ii = threadIdx.x + blockDim.x*blockIdx.x, tz = ii / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
	unsigned int oz = offset / (x*y), oy = (offset % (x*y)) / x, oz = (offset % (x*y)) % x;
	for (int i = ii; i < Ux->size && Uy->size && Uz->size && i < divU->size; i += offset) {
		divU[i] -= (Ux->d_a[i] + Uy->d_a[i] + Uz->d_a[i]);
		if (tx > 0 && Ux->d_i[i - 1] == Ux->d_i[i] - 1) {
			divU->d_a[i] += Ux->d_a[i - 1];
		}
		if (ty > 0 && Uy->d_i[i - x] == Uy->d_i[i] - x) {
			divU->d_a[i] += Uy->d_a[i - x];
		}
		if (tz > 0 && Uz->d_i[i - x * y] == Uz->d_i - x*y) {
			divU->d_a[i] += Uz->d_a[i - x * y];
		}

		tz += oz;
		ty += oy;
		while (ty >= y) {
			ty -= y;
			++tz;
		}
		tx += ox;
		if (tx >= x) {
			tx -= x;
			++ty;
			if (ty >= y) {
				ty -= y;
				++tz;
			}
		}
	}
}

template <typename T>
__global__ void maxResidual(CuVec<T>* in) {
	
}

template <typename T>
__device__ void TriVec<T>::updateU(){
	int offset = blockDim.x*gridDim.x;
	for(int i = threadIdx.x+blockIdx.x*blockDim.x; i < size; i += offset){
		if(a[i].t != SOLID){
			T p = a[i].p;
			int tz = i/(x*y), ty = (i%(x*y))/x, tx = (i%(x*y))%x;
			T dpx = 0, dpy = 0, dpz = 0;
			if(get(tx+1, ty, tz).t != SOLID)
				dpx = get(tx+1, ty, tz).p - p;

			if(get(tx, ty+1, tz).t != SOLID)
				dpy = get(tx, ty+1, tz).p - p;

			if(get(tx, ty, tz+1).t != SOLID)
				dpz = get(tx, ty, tz+1).p - p;

			Vec3<T> t(dpx, dpy, dpz);	//calculate pressure diffs in x,y,z
			t = t*dt/dx/density;		//scale these pressure diffs
			a[i].u = a[i].u - t;
		}
	}
}

template <typename T>
void TriVec<T>::updateUCPU() {
	int offset = 1;
	T scale = dt / (dx*density);
//#pragma omp parallel for
	//for (int i = 0; i < size; i += offset) {
	concurrency::parallel_for(size_t(0), size_t(size), [&](int i){
		if (a[i].t != SOLID) {
			//if(a[i].t == FLUID){
			int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
			T dpx, dpy, dpz;
			if (get(tx + 1, ty, tz).t != SOLID)
				//if(get(tx+1, ty, tz).t == FLUID)
				dpx = get(tx + 1, ty, tz).p - get(tx, ty, tz).p;
			else dpx = 0;

			if (get(tx, ty + 1, tz).t != SOLID)
				//if (get(tx, ty+1, tz).t == FLUID)
				dpy = get(tx, ty + 1, tz).p - get(tx, ty, tz).p;
			else dpy = 0;

			if (get(tx, ty, tz + 1).t != SOLID)
				//if (get(tx, ty, tz+1).t == FLUID)
				dpz = get(tx, ty, tz + 1).p - get(tx, ty, tz).p;
			else dpz = 0;

			Vec3<T> t(dpx, dpy, dpz);	//calculate pressure diffs in x,y,z
			t = t * scale;
			a[i].u = a[i].u - t;																				//subtract pressure-derived term per equation 43
		}
	});
}

template <typename T>
Vec3<T> TriVec<T>::negA(int x, int y, int z){
	return Vec3<T>(get(x-1, y, z).aX, get(x, y-1, z).aY, get(x, y, z-1).aZ);
}

#define W 1.5

template <typename T>
void TriVec<T>::singleThreadGS(){
	concurrency::parallel_for(size_t(0), size_t(size), [&](int i){
		if (a[i].t == FLUID) {
			int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
			a[i].p = a[i].p + W * ((a[i].divU - (a[i].aX*get(tx + 1, ty, tz).p + a[i].aY*get(tx, ty + 1, tz).p + a[i].aZ*get(tx, ty, tz + 1).p + a[i].anX*get(tx - 1, ty, tz).p + a[i].anY*get(tx, ty - 1, tz).p + a[i].anZ*get(tx, ty, tz - 1).p)) / (a[i].aDiag + 0.000000000000001) - a[i].p);
		}
		else a[i].p = 0;
	});
}

template <typename T>
__device__ void TriVec<T>::calcResidualGS(){
	int offset = gridDim.x*blockDim.x;
	for(int i = threadIdx.x + blockDim.x*blockIdx.x; i < size; i += offset){
		if(a[i].t == FLUID){
			int tz = i/(x*y), ty = (i%(x*y))/x, tx = (i%(x*y))%x;						//get x, y, z coords of voxel. Calculate residual of pressure step for each fluid Voxel
			Vec3<T> t = negA(tx, ty, tz);
			res[i] = a[i].divU - (a[i].aDiag*a[i].p + (a[i].aX*get(tx+1, ty, tz).p + a[i].aY*get(tx, ty+1, tz).p + a[i].aZ*get(tx, ty, tz+1).p + t.x*get(tx-1, ty, tz).p + t.y*get(tx, ty-1, tz).p + t.z*get(tx, ty, tz-1).p));
		}
		a[i].pold = a[i].p;
	}
}

template <typename T>
void TriVec<T>::calcResidualCPU() {
	concurrency::parallel_for(size_t(0), size_t(size), [&](int i){
		if (a[i].t == FLUID) {
			int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;						//get x, y, z coords of Voxel. Calculate residual of pressure step for each fluid Voxel
			res[i] = a[i].divU - (a[i].aDiag*a[i].p + (a[i].aX*get(tx + 1, ty, tz).p + a[i].aY*get(tx, ty + 1, tz).p + a[i].aZ*get(tx, ty, tz + 1).p + a[i].anX*get(tx - 1, ty, tz).p + a[i].anY*get(tx, ty - 1, tz).p + a[i].anZ*get(tx, ty, tz - 1).p));
		}
		else res[i] = 0;
	});
}

template <typename T>
__device__ void TriVec<T>::multiThreadJacobi(T* ax){
	long long index = threadIdx.x + blockDim.x*blockIdx.x;
	int offset = gridDim.x*blockDim.x;
	for(int i = index; i < size; i += offset){									//each thread searches for a fluid cell so all run into a fluid unless there are more threads than fluid cells
		int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
		if(a[i].t == FLUID){
			Voxel<T> t = a[i];
			a[i].p = (t.divU - (t.aX*get(tx+1, ty, tz).pold + t.aY*get(tx, ty+1, tz).pold + t.aZ*get(tx, ty, tz+1).pold + t.anX*get(tx-1, ty, tz).pold + t.anY*get(tx, ty-1, tz).pold + t.anZ*get(tx, ty, tz-1).pold))/(a[i].aDiag+0.000000000000001);
		}
		else {
			a[i].p = 0;
		}
	}
}


template <typename T>
__device__ void TriVec<T>::redSOR(){
	int offset = (gridDim.x*blockDim.x) << 1;
	for (unsigned long long i = (threadIdx.x + blockDim.x*blockIdx.x) << 1; i < numF; i += offset) {
		double t = 0, adiag;
		for (int j = A.d_r[i]; j < A.d_r[i + 1]; ++j) {
			if(A.c[j] != i)
				t += A.a[j] * pressure;
			else {
				adiag = A.a[j];
			}
		}
		pressure[i] = pressure_old[i] + W * (t / (adiag + 0.000000000000001) - pressure_old[i]));
	}
}

template <typename T>
__device__ void TriVec<T>::blackSOR() {
	int offset = (gridDim.x*blockDim.x) << 1;
	for (unsigned long long i = ((threadIdx.x + blockDim.x*blockIdx.x) << 1) + 1; i < numF; i += offset) {
		double t = 0, adiag;
		for (int j = A.d_r[i]; j < A.d_r[i + 1]; ++j) {
			if (A.d_c[j] != i)
				t += A.d_a[j] * pressure[A.d_c[j]];
			else {
				adiag = A.d_a[j];
			}
		}
		pressure[i] = pressure_old[i] + W * (t / (adiag + 0.000000000000001) - pressure_old[i]);
	}
}

template <typename T>
__device__ void TriVec<T>::maxResidual(T* s, T* glob){	//trying different parallel implementations to speed up max search.
	int index = threadIdx.x + blockIdx.x*blockDim.x, offset = gridDim.x*blockDim.x;
	s[threadIdx.x] = 0;
	for(int i = index; i < size; i+=offset){
		if (abs(res[i]) > abs(s[threadIdx.x]))
			s[threadIdx.x] = abs(res[i]);
	}
	__syncthreads();
	for(int i = blockDim.x/2; i > 0; i>>=1){
		if(threadIdx.x < i)
			if(s[threadIdx.x+i] > s[threadIdx.x])
				s[threadIdx.x] = s[threadIdx.x +i];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		glob[blockIdx.x] = s[0];
}

 void threadedMaxResidual(double* res, int size, int threadNum, int numThreads, double* curMax) {
	 *curMax = 0;
	 int maxIndex = size / numThreads;
	 int curIndex = threadNum * (maxIndex);
	 for (int i = 0; i < maxIndex && i + curIndex < size; ++i) {
		 if (abs(res[i + curIndex]) > *curMax) {
			 *curMax = abs(res[i]);
		 }
	 }
 }

template <typename T>
void TriVec<T>::maxResidualCPU() {
	maxRes = 0;
	unsigned int NUMT = std::thread::hardware_concurrency();
	
	T* maxes = new T[NUMT];
	std::future<void>** t = new std::future<void>*[NUMT];
	for (int i = 0; i < NUMT; ++i) {
		t[i] = new future<void>;
		*t[i] = std::async(std::launch::async, threadedMaxResidual, this->res, size, i, NUMT, &maxes[i]);
	}
	for (int i = 0; i < NUMT; ++i) {
		t[i]->get();
		if (maxes[i] > maxRes)
			maxRes = maxes[i];
		delete t[i];
	}
	delete[] maxes;
	delete[] t;

}

template <typename T>
__device__ void TriVec<T>::maxU(T* glob){
	int index = threadIdx.x + blockIdx.x*blockDim.x, offset = gridDim.x*blockDim.x;
	T curMax = 0;
	for(int l = index; l < size; l+=offset){
		if(a[l].t == FLUID){
			int k = l / (x*y), j = (l % (x*y)) / x, i = (l % (x*y)) % x;
			T t = sqrt(a[l].u.x*a[l].u.x + a[l].u.y*a[l].u.y + a[l].u.z*a[l].u.z);
			if(t > curMax)
				curMax = t;
		}
	}
	glob[index] = curMax;
	__syncthreads();
	for(int i = blockDim.x/2; i > 0; i>>=1){
		if(index < i)
			if(glob[index+i] > glob[index])
				glob[index] = glob[index+i];
		__syncthreads();
	}
	if(index == 0) mU = glob[0];
}

template <typename T>
__device__ void TriVec<T>::copyFrom(TriVec<T>& in){
	int offset = gridDim.x*blockDim.x;
	for(int i = threadIdx.x + blockIdx.x*blockDim.x; i < in.size; i+=offset){
		a[i] = in.a[i];
		
		if(a[i].t == FLUID){
			a[i].copyData(in.a[i]);
		}
	}
}

template <typename T>
void TriVec<T>::setDimensions(int x, int y, int z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->size = x*y*z;
	
	if (a != nullptr) {
		delete[] a;
		a = new Voxel<T>[size];
	}
	if (res != nullptr) {
		delete[] res;
		res = new T[size];
	}
	if (pFluidIndexes != nullptr) {
		delete[] pFluidIndexes;
		pFluidIndexes = new int[size];
	}
}

template <typename T>
void TriVec<T>::setdx(double nx) {
	this->dx = nx;
}

template <typename T>
__device__ void TriVec<T>::removeAandLists(){
	int offset = gridDim.x*blockDim.x;
	if (threadIdx.x + blockDim.x*blockIdx.x == 0) {
		numPFluids = 0;
	}
	for(int i = threadIdx.x+blockDim.x*blockIdx.x; i < size; i += offset){
		a[i].t = EMPTY;
		a[i].p = a[i].pold = 0;
		a[i].u = Vec3<T>();
		a[i].uOld = Vec3<T>();
		a[i].aDiag = 0;
		a[i].aX = 0;
		a[i].aY = 0;
		a[i].aZ = 0;
		a[i].divU = 0;
		res[i] = 0;
		a[i].pureFluid = false;
		a[i].sumk = Vec3<T>();
		a[i].sumuk = Vec3<T>();
	}
}

template <typename T>
void TriVec<T>::resetFluidsCPU() {
	concurrency::parallel_for(size_t(0), size_t(size), [&](int i){
		a[i].u = Vec3<T>();
		a[i].uOld = Vec3<T>();
		a[i].t = EMPTY;
	});
}

template <typename T>
void TriVec<T>::applyBodyForces() {
	Vec3<T> t = Vec3<T>(0,0,-9.8) * dt;
#ifdef __CUDA_ARCH__
	int offset = gridDim.x*blockDim.x;
	for (int l = threadIdx.x + blockIdx.x*blockDim.x; l < size; l += offset) {
		if (a[l].t != SOLID) {
			a[l].u += t;
		}
	}
#else
	concurrency::parallel_for(size_t(0), size_t(size), [&](int l){
		if (a[l].t != SOLID) {
			a[l].u += t;
		}
	});
#endif
}

template <typename T>
void TriVec<T>::applyBodyForcesCPU() {
	concurrency::parallel_for(size_t(0), size_t(size), [&](int l){
		if (a[l].t != SOLID) {
			a[l].u += t;
		}
	});
}


template <typename T>
__device__ void TriVec<T>::findPureFluids() {	//build linked list of fluid voxels from this->a that are completely surrounded (on the neighboring 6 faces) by fluid cells
	int offset = gridDim.x*blockDim.x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < size; i += offset) {
		if (a[i].t == FLUID) {
			int tx = (i % (x*y)) % x, ty = (i % (x*y)) / x, tz = (i / (x*y));
			if (get(tx - 1, ty, tz).t == FLUID && get(tx, ty - 1, tz).t == FLUID && get(tx, ty, tz - 1).t == FLUID && get(tx + 1, ty, tz).t == FLUID && get(tx, ty + 1, tz).t == FLUID && get(tx, ty, tz + 1).t == FLUID) {
				a[i].pureFluid = true;
			}
		}
	}
}

template <typename T>
void TriVec<T>::findPureFluidsCPU() {
	numPFluids = 0;
	for (int i = 0; i < size; ++i) {
		if (a[i].t == FLUID) {
			int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
			if (get(tx + 1, ty, tz).t == FLUID && get(tx - 1, ty, tz).t == FLUID)
				if (get(tx, ty + 1, tz).t == FLUID && get(tx, ty - 1, tz).t == FLUID)
					if (get(tx, ty, tz + 1).t == FLUID && get(tx, ty, tz - 1).t == FLUID)
						pFluidIndexes[numPFluids++] = i;
		}
	}
	if (!numPFluids) {
		for (int i = 0; i < size; ++i) {
			if (a[i].t == FLUID) {
				pFluidIndexes[numPFluids++] = i;
			}
		}
	}
}

template <typename T>
__device__ void TriVec<T>::constructPureFluidList() {
	int offset = gridDim.x*blockDim.x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < size; i += offset) {
		//if (a[i].t == FLUID) {
		if(a[i].pureFluid){
			int l = atomicAdd(&numPFluids, 1);
			pFluidIndexes[l] = i;
		}
	}
}

template <typename T>
double TriVec<T>::pWeight(Vec3<T> in) {
	in = in / dx;
	return 1 / sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

template <typename T>
__device__ double TriVec<T>::pWeight(const T& inx, const T& iny, const T& inz, const T& dxdx) {
	return 1 / sqrt(dxdx / (inx*inx) + dxdx / (iny*iny) + dxdx / (inz*inz));
}

template <typename T>
__device__ void TriVec<T>::insertToRandomPureFluid(Particle<T>& in, curandState_t* states) {
	curandState_t* st = states + threadIdx.x + blockIdx.x*blockDim.x;
	int voxel = pFluidIndexes[(int)(curand_uniform(st) * numPFluids)];
 	int tx = (voxel % (x*y)) % x, ty = (voxel % (x*y)) / x, tz = (voxel / (x*y));
	if (!a[voxel].pureFluid) {
		printf("NOT A PURE FLUID VOXEL\n");
	}
	in.p = Vec3<T>(tx*dx + curand_uniform(st)*dx, ty*dx + curand_uniform(st)*dx, tz*dx + curand_uniform(st)*dx);
	interpUtoP(in);
}

template <typename T>
void TriVec<T>::insertToRandomPureFluidCPU(Particle<T>& in) {
	int i = pFluidIndexes[(int)(dist->operator()(*gen)*numPFluids)];
	int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
	in.p.x = tx * dx + dist->operator()(*gen)*dx;
	in.p.y = ty * dx + dist->operator()(*gen)*dx;
	in.p.z = tz * dx + dist->operator()(*gen)*dx;
	interpUtoP(in);
}