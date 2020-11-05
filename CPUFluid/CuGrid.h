#ifndef CUGRID_H
#define CUGRID_H
#include <fstream>
#include <string>
#include "TriVec.h"
using namespace Cu;

namespace Cu {

	template <typename T>
	class CuGrid {
	public:
		CuGrid();
		CuGrid(int x, int y, int z, double dx, double dt, double density);
		CuGrid(const CuGrid&);
		~CuGrid();
		void applyU();
		void interpU();
		void advectParticles();
		void initializeParticles();
		void setTstep(double t);
		void print();
		void maxU();
		void printParts(string filename);
		const CuGrid& operator=(const CuGrid&);
		void setFluidCells();
		void reinsertOOBParticles();
		TriVec<T> a;
		unsigned long long x, y, z;
		double dx, dt, density;
		vector<Particle<T>> particles;
		
	private:
		void nullifyHostTriVecs();
		void nullifyDeviceTriVecs();
	};

#include "CuGrid.hpp"
}

#endif