#ifndef VOXEL_H
#define VOXEL_H
#include "Vec3.h"
#include "Particle.h"
namespace Cu{
    enum VoxType {EMPTY, SOLID, FLUID};
	template <typename T>
	class Voxel{
	public:
		__device__ __host__ Voxel();
		__device__ __host__ Voxel(bool in);
		__device__ __host__ Voxel(const Voxel& in);
		__device__ __host__ const Voxel& operator=(const Voxel& in);
		__device__ __host__ void removeParticle(int index);
		Vec3<T> u, uOld;
		T p, pold, divU, sd;
		VoxType t;
		bool invalid, pureFluid;
		Vec3<T> sumk, sumuk;
		T aDiag, aX, aY, aZ;
		T anX, anY, anZ;
	};
    #include "Voxel.hpp"
}
#endif