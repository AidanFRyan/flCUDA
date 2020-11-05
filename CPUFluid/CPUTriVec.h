#ifndef CPUTriVec_H
#define CPUTriVec_H

#include "Voxel.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
// #include <vector>
using namespace Cu;
namespace Cu {
	template <typename T>
	class CPUTriVec {
	public:
		CPUTriVec();
		CPUTriVec(unsigned long long x, unsigned long long y, unsigned long long z, double dx, double dt, double density, bool dev = false);
		const CPUTriVec<T>& operator=(const CPUTriVec& in);
		T pdx(int x, int y, int z);
		T pdy(int x, int y, int z);
		T pdz(int x, int y, int z);
		T udx(int x, int y, int z);
		T udy(int x, int y, int z);
		T udz(int x, int y, int z);
		Vec3<T> backsolveU(T x, T y, T z, T dt);
		void interpU();
		Vec3<T> laplacianU(int x, int y, int z);
		CPUVoxel<T>& get(int x, int y, int z);
		const CPUTriVec<T>& operator=(CPUTriVec&& in);
		~CPUTriVec();
		T kWeight(Vec3<T> x);
		T divergenceU(int x, int y, int z);
		void maxResidual();	//synchronous for block
		void applyA();
		void dotZS();
		void dotZR();
		void updateP(double a);
		void updateR(double a);
		void updateS(double b);
		void updateZ();
		void updateU();
		//void applyU();
		void advectParticles();
		void singleThreadGS();
		void calcResidualGS();
		Vec3<T> negA(int x, int y, int z);
		//void maxU();
		void copyFrom(CPUTriVec<T>& in);
		void fillGaps();
		void interpUtoP(Particle<T>& in);
		void multiThreadJacobi();
		void findPureFluids();
		void reinsertToFluid(Particle<T>& in);
		void resetFluids();

		std::default_random_engine gen;
		std::uniform_real_distribution<double> dist;
		std::vector<int> pureFluidList;
		CPUVoxel<T>* a;
		T dx, dt;
		unsigned long long x, y, z, size;
		CPUVoxel<T> invalid;
		T maxRes, density;
		T mU;
	private:
		T pd(const T& l, const T& v, const T& r);
	};
#include "CPUTriVec.hpp"
}
#endif