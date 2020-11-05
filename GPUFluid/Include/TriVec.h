#ifndef TRIVEC_H
#define TRIVEC_H
#include "Voxel.h"
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <iostream>
#include <future>
#include "SLA.h"
//#include <thread>

namespace Cu {

	template <typename T>
	class TriVec {//was originally supposed to be a way to store three arrays for each component of velocity, each element being a voxel to which the component corresponded
					//need to refactor back into just arrays
	public:
		__device__ __host__ TriVec();
		__device__ __host__ TriVec(unsigned long long x, unsigned long long y, unsigned long long z, double dx, double dt, double density, bool dev = false);
		__device__ __host__ ~TriVec();
		__device__ __host__ const TriVec<T>& operator=(const TriVec& in);
		//__device__ void weightedAverage(const int& v, const int& i, const int& j, const int& k);
		__device__ __host__ void interpU();
		//__device__ void interpU(T* sumukx, T* sumuky, T* sumukz, T* sumkx, T* sumky, T* sumkz, T* t, T* cPxx, T* cPxy, T* cPxz, T* cPyx, T* cPyy, T* cPyz, T* cPzx, T* cPzy, T* cPzz, T* dxdx);
		__device__ __host__ Voxel<T>& get(int x, int y, int z);
		__device__ __host__ const TriVec<T>& operator=(TriVec&& in);
		__device__ __host__ T kWeight(Vec3<T> x);
		__device__ __host__ T divergenceU(int x, int y, int z);
		__device__ void maxResidual(T* glob, T* s);
		void maxResidualCPU();
		__device__ void updateU();
		void updateUCPU();
		void singleThreadGS();
		__device__ void calcResidualGS();
		void calcResidualCPU();
		__device__ __host__ Vec3<T> negA(int x, int y, int z);
		__device__ void maxU(T* glob);
		__device__ void copyFrom(TriVec<T>& in);
		__device__ void fillGaps();
		//__device__ void interpUtoP(Particle<T>& in, T* ux, T* uy, T* uz, T* uOldx, T* uOldy, T* uOldz);
		__device__ __host__ void interpUtoP(Particle<T>& in);
		__device__ void multiThreadJacobi(T* ax);
		__device__ __host__ void applyBodyForces();
		void applyBodyForcesCPU();

		__device__ void findPureFluids();
		void findPureFluidsCPU();
		__device__ void constructPureFluidList();	//single threaded
		__device__ void insertToRandomPureFluid(Particle<T>& in, curandState_t* states);	//to be run as part of advectParticles step
		void insertToRandomPureFluidCPU(Particle<T> &in);
		__device__ __host__ double pWeight(Vec3<T> in);
		__device__ double pWeight(const T& inx, const T& iny, const T& inz, const T& dxdx);
		__device__ void redSOR();
		__device__ void blackSOR();

		void setDimensions(int x, int y, int z);
		void setdx(double nx);

		__device__ void removeAandLists();
		void resetFluidsCPU();

		Voxel<T>* a;
		T* res;
		int *pFluidIndexes, numPFluids;
		T dx, dt;
		T invalidX, invalidY, invalidZ;
		unsigned long long x, y, z, size;
		Voxel<T> invalid;
		int* solids, numSolids;
		T maxRes, density;
		T mU;

		std::default_random_engine* gen;
		std::uniform_real_distribution<double>* dist;

		SLA::CuMat<double> A;
		SLA::CuVec<double> pressure, pressure_old, div_U, uX, uY, uZ, residual;
		SLA::CuVec<VoxType> types;
		int numF;
	};

#include "TriVec.hpp"
}

#endif