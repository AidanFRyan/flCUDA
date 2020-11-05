#include "CuSolver.h"
#define FRAMES 330
#define NUMCELLS 10
#define FPS 30
#define DX 1
using namespace Cu;

int main() {
	CuSolver<double> solver(NUMCELLS, NUMCELLS, NUMCELLS, FRAMES, DX, 1.0/FPS);
	solver.testInit();
	solver.advect();
	return 0;
}
