main: TriVec.h* CuGrid.h* CuSolver.h* Vec3.h* Particle.h* CuDriver.cpp Makefile
	g++ -Wall -std=c++11 -mavx2 CuDriver.cpp -o main

nonavx: TriVec.h* CuGrid.h* CuSolver.h* Vec3.h* Particle.h* CuDriver.cpp Makefile
	g++ -Wall -std=c++11 CuDriver.cpp -o nonavx

debug: TriVec.h* CuGrid.h* CuSolver.h* Vec3.h* Particle.h* CuDriver.cpp Makefile
	g++ -g CuDriver.cpp -o main
