#ifndef CUSOLVER_H
#define CUSOLVER_H

#include <vector>
#include "CuGrid.h"
#include <iostream>
using namespace std;
using namespace Cu;

namespace Cu{
	template <typename T>	//T should only be T or double
	class CuSolver {
	public:
		CuSolver(int x = 0, int y = 0, int z = 0, int frames = 0, double dx = 0,  double dt = 0);
		~CuSolver();
		bool initialValues(CuGrid<T>* initial);
		bool setSolidCells(int frame, vector<T>& initial);
		bool solve();
		bool advect();	//single frame advect, stores to next index, should be called by solve function, need to evaluate copy every frame to and from cpu/gpu
		void resize(int x, int y, int z);
		void resizeTime(int t);
		void testInit();
		void printGrid(int frame);
		void printParts(int frame);
		void initializeParticles();
	private:
		CuGrid<T>* targetGrid, *sourceGrid;	//grid for each frame in simulation
		int x, y, z, frames, lastSolved;	//x, y, z dimensions of grid and number of frames
		double dt, dx;	//dt per frame, dx per cell (cube cells)
	};
#include "CuSolver.hpp"
}

#endif