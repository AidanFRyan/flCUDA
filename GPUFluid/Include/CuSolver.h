#ifndef CUSOLVER_H
#define CUSOLVER_H
#include <tbb/tbb.h>

#include <vector>
#include "Kernels.h"
#include "Triangle.h"
//#include "GPUFluid/Kernels.h"
#include <iostream>
using namespace std;



namespace Cu{
	template <typename T>	//T should only be float or double
	class CuSolver {
	public:
		CuSolver(double dimx, double dimy, double dimz, int frames, double dx,  double dt);
		~CuSolver();
		bool initialValues(CuGrid<T>* initial);
		bool setSolidCells(int frame, vector<T>& initial);
		bool solve();
		bool advect();	//single frame advect, stores to next index, should be called by solve function, need to evaluate copy every frame to and from cpu/gpu
		void resize(int x, int y, int z);
		void resizeTime(double t);
		void testInit();
		void printGrid(int frame);
		void printParts(int frame);
		void setDimensions(double x, double y, double z);
		void setFrames(int f);
		void setFPS(double fps);
		void readParticlesFromTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p);
		void writeParticlesToTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p);
		void solveFrame(CuGrid<T>*);
		void advectSingleFrameGPU();
		void advectSingleFrameCPU();
		void setdx(double newX);
		void setWorldPosition(double x, double y, double z);	//set world position of grid's local 0,0,0
		void setSubSamples(int n);

		void voxelizeSolids(Triangle<T>* in, unsigned int numTris);
	private:
		CuGrid<T>* grid;	//grid for each frame in simulation
		int x, y, z, frames, lastSolved;	//x, y, z dimensions of grid and number of frames
		Vec3<T> worldPos;
		double dt, dx, dimx, dimy, dimz;	//dt per frame, dx per cell (cube cells)
		curandState_t *states;
		HINSTANCE vComp;
	};
#include "CuSolver.hpp"
}

#endif