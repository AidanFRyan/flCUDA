#include "CPUSolver.h"
#define FRAMES 2
#define NUMCELLS 100
#define FPS 24
#define DX 0.1
using namespace Cu;

int main() {
	CPUSolver<double> solver(NUMCELLS, NUMCELLS, NUMCELLS, FRAMES, DX, 1.0/FPS);
	solver.testInit();
	solver.advect();
	return 0;
}