#include "Include/CPUFluid/CPUSolver.h"
#define FRAMES 360
#define NUMCELLS 40
#define FPS 24
#define DX 0.25
using namespace Cu;

int main() {
	CPUSolver<double> solver(NUMCELLS, NUMCELLS, NUMCELLS, FRAMES, DX, 1.0/FPS);
	solver.testInit();
	solver.advect();
	return 0;
}
