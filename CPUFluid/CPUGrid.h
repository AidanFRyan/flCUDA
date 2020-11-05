#ifndef CPUGrid_H
#define CPUGrid_H
#include <fstream>
#include <string>
#include "CPUTriVec.h"
using namespace Cu;

namespace Cu {

	template <typename T>
	class CPUGrid {
	public:
		CPUGrid();
		CPUGrid(int x, int y, int z, double dx, double dt, double density);
		CPUGrid(const CPUGrid&);
		~CPUGrid();
		void applyU();
		void interpU();
		void advectParticles();
		void initializeParticles();
		void setTstep(double t);
		void print();
		void maxU();
		void printParts(string filename);
		const CPUGrid& operator=(const CPUGrid&);
		void setFluidCells();
		void reinsertOOBParticles();
		CPUTriVec<T> a;
		unsigned long long x, y, z;
		double dx, dt, density;
		vector<Particle<T>> particles;
		
	private:
		void nullifyHostCPUTriVecs();
		void nullifyDeviceCPUTriVecs();
	};

#include "CPUGrid.hpp"
}

#endif