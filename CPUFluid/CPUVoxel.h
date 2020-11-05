#ifndef CPUVoxel_H
#define CPUVoxel_H
#include "CPUParticle.h"
using namespace Cu;
namespace Cu{
    enum VoxType {SOLID, FLUID, EMPTY};
	template <typename T>
	class CPUVoxel{
	public:
		CPUVoxel();
		CPUVoxel(bool in);
		CPUVoxel(const CPUVoxel& in);
		const CPUVoxel& operator=(const CPUVoxel& in);
		void applyBodyForces(T dt);
		Vec3<T> u, uOld;	//note that f is actually acceleration vector in this version of the code
		T p, divU, res;	//pressure, divergence of u term (for pressure solver), residual, z vector quantities (negative residual in CG, preconditioned value in PCG)
		VoxType t;
		bool invalid;
		T aDiag, aX, aY, aZ;//, dotTemp;
		Vec3<T> sumuk, sumk;	//these don't really need to be stored inside CPUVoxels, try and use temp variables for this part of computation
	};
    #include "CPUVoxel.hpp"
}
#endif