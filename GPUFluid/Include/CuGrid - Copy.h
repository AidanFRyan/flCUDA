#ifndef CUGRID_H
#define CUGRID_H
#include <fstream>
#include <string>
#include "TriVec.h"
#include "Particle.h"

namespace Cu {

	template <typename T>
	class CuGrid {
	public:
		__device__ __host__ CuGrid();
		CuGrid(unsigned int x, unsigned int y, unsigned int z, double dx, double dt, double density);
		__device__ __host__	CuGrid(const CuGrid&);
		__host__ ~CuGrid();
		CuGrid* toDeviceCopy();
		CuGrid* toDeviceEmpty();
		bool copyFromDevice(CuGrid* d_grid);
		bool copyFromDeviceAsync(CuGrid* d_grid, cudaStream_t& stream);
		bool allocateOnDevice();
		void removeDeviceCopies();
		__device__ __host__ void setTstep(double t);
		void print();
		void printParts(string filename);
		void initializeList(int n);
		__device__ __host__ const CuGrid& operator=(const CuGrid&);
		void removeDeviceTriVec();
		void setDimensions(int x, int y, int z);
		void readParticlesFromTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p, const Vec3<T>& offsetP);
		void writeParticlesToTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p, const Vec3<T>& offsetP);
		__device__ void advectParticles();
		void advectParticlesCPU();
		__device__ void reInsert(curandState_t* states);
		void reInsertCPU();
		__device__ void construct();
		void constructCPU();
		__device__ void applyU();
		void applyUCPU();
		__device__ void interpU(T* sumukx, T* sumuky, T* sumukz, T* sumkx, T* sumky, T* sumkz);
		__device__ void interpU();
		void interpUCPU();
		void setdx(double nx);
		void setSubSamples(int n);

		void maxUCPU();

		TriVec<T> a, d_a;
		int x, y, z, subSamples;
		T dx, dt, density;
		Particle<T>* list, *d_list;
		int numParticles;
	private:
		void nullifyHostTriVecs();
		void nullifyDeviceTriVecs();
	};

#include "CuGrid.hpp"
}

#endif