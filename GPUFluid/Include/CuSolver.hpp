#include "CuSolver.h"
//#include <unistd.h>
using namespace Cu;

template <typename T>
CuSolver<T>::CuSolver(double dimx, double dimy, double dimz, int frames, double dx, double t) {
	//omp_set_num_threads(omp_get_max_threads());
	//int aa = omp_get_num_procs();
	//printf("%d\n", aa);
	this->dimx = dimx;
	this->dimy = dimy;
	this->dimz = dimz;
	/*if (dx <= 0 && dimx <= 0 && dimy <= 0) {
		dx = 0.1;
		dimx = 1.0;
		dimy = 1.0;
		dimz = 1.0;
	}*/
	this->x = (int)(dimx / dx);
	this->y = (int)(dimy / dx);
	this->z = (int)(dimz / dx);
	if (!x)
		x = 1;
	if (!y)
		y = 1;
	if (!z)
		z = 1;
	this->dx = dx;
	this->frames = frames;
	this->dt = t;
	lastSolved = -1;
	grid = new CuGrid<T>(x, y, z, dx, t, 997);	//density of water is 997 kg/m^3 @ 20c
	//cout << "set dt to " << dt << "in grid\n";
	cudaMalloc((void**)&states, sizeof(curandState_t)*NUMBLOCKS*NUMTHREADS);
	cudaDeviceSynchronize();
	initStates << <NUMBLOCKS, NUMTHREADS >> > (states);
	cudaDeviceSynchronize();
}

template <typename T>
CuSolver<T>::~CuSolver() {
	if (states != nullptr) {
		cudaFree(states);
	}
	if (grid != nullptr) {
		delete grid;
	}
}

template <typename T>
void CuSolver<T>::testInit(){
	int numP = 0, numS = 0;
	for(int k = 0; k < z; k++){
		for(int j = 0; j < y; j++){
			for(int i = 0; i < x; i++){
				Voxel<T>* t = &grid->a.get(i, j, k);
				if(i == 0 || i == x-1 || j == 0 || j == y-1 || k == 0 || k == z-1){
					// printf("%p %d %d %d solid\n", t, i, j, k);
					t->t = SOLID;
					++numS;
				}
				else if(j < 5*y/8 && j > 3*y/8){
					t->t = FLUID;
				// else if(k > 2*z/3 && k < 7*z/8 && j >= 3*y/8 && j <= 5*y/8 && i >= 3*x/8 && i <= 5*x/8){
				// // else if(k > z/2){
				// 	// printf("%p %d %d %d fluid\n", t, i, j, k);
				// 	t->t = FLUID;
				// }
					numP += 8;
				}
				else{
					// printf("%p %d %d %d empty\n", t, i, j, k);
					t->t = EMPTY;
				}
			}
		}
	}
	this->grid->initializeList(numP);
	this->grid->d_a.numSolids = numS;
	this->grid->a.numSolids = numS;
	cout<<"numSolids "<<this->grid->d_a.numSolids<<endl;
}

template <typename T>
bool CuSolver<T>::initialValues(CuGrid<T>* initial) {
	if (!frames ||	initial->x*initial->y*initial->z != x*y*z) return false;
	grid = initial;
	lastSolved = 0;
	return true;
}

template <typename T>
void CuSolver<T>::setFrames(int f) {
	this->frames = f;
}

template <typename T>
void CuSolver<T>::setFPS(double fps) {
	this->dt = 1.0 / fps;
	this->grid->dt = this->dt;
	this->grid->a.dt = this->dt;
}

template <typename T>
void CuSolver<T>::solveFrame(CuGrid<T>* initialDevGrid) {
	printf("initialDevGrid %p\n", initialDevGrid);
	T* glob, max, *h_max, *d_max, *g;
	double delt;
	gpuErrchk(cudaMalloc((void**)&glob, NUMBLOCKS*NUMTHREADS * sizeof(T)));
	gpuErrchk(cudaMalloc((void**)&d_max, sizeof(T)));
	g = new T[NUMBLOCKS*NUMTHREADS];
	h_max = new T;
	cout<<"cusolver dt: "<<dt<<" cugrid dt: "<<grid->dt<<endl;
	(resetGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
	gpuErrchk(cudaDeviceSynchronize());
	(constructGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));	//major slow, but takes care of multiple writes to same linked list. Need to parallelize (KD Tree for particle sorting)
	gpuErrchk(cudaDeviceSynchronize());
	(findPureFluids << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
	gpuErrchk(cudaDeviceSynchronize());
	constructPureFluidList << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid);
	gpuErrchk(cudaDeviceSynchronize());
	(reInsertSolids << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid, states));
	gpuErrchk(cudaDeviceSynchronize());
	for (T tElapsed = 0; grid->dt - tElapsed > 0.00001;) {
		printf("initialDevGrid %p\n", initialDevGrid);
		cout<<tElapsed<<" "<<grid->dt<<endl;
		(interpolateFromParts << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(interpolateToGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		/*interpolateToGrid << <NUMBLOCKS, NUMTHREADS, (16*NUMTHREADS+1)*sizeof(T) >> > (initialDevGrid);
		cudaDeviceSynchronize();*/
		(copyUOld << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(getMaxU << <1, NUMTHREADS >> > (initialDevGrid, glob, d_max));
		gpuErrchk(cudaDeviceSynchronize());
		(setParameters << <1, 1 >> > (initialDevGrid, grid->dt, tElapsed, d_max));
		gpuErrchk(cudaDeviceSynchronize());
		(cudaMemcpy(h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost));
		delt = *h_max;
		gpuErrchk(cudaDeviceSynchronize());
		(bodyForces << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(divergence << <NUMBLOCKS, NUMTHREADS, NUMTHREADS*sizeof(T) >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(getMaxResidual << <NUMBLOCKS, NUMTHREADS, NUMTHREADS * sizeof(T) >> > (initialDevGrid, d_max));
		gpuErrchk(cudaDeviceSynchronize());
		(cudaMemcpy(g, glob, NUMBLOCKS * sizeof(T), cudaMemcpyDeviceToHost));
		max = 0;
		for (int i = 0; i < NUMBLOCKS; ++i) {
			if (g[i] > max) {
				max = g[i];
			}
		}
		for (int j = 0; j < 10000 && max > 0.00000001; ++j) {
			/*jacobi << <NUMBLOCKS, NUMTHREADS, (NUMTHREADS+1)*sizeof(T) >> > (initialDevGrid);
			cudaDeviceSynchronize();*/
			(redSOR << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
			gpuErrchk(cudaDeviceSynchronize());
			(blackSOR << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
			gpuErrchk(cudaDeviceSynchronize());
			/*asyncSOR<<<NUMBLOCKS, NUMTHREADS>>>(initialDevGrid);
			cudaDeviceSynchronize();*/
			(residual << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
			gpuErrchk(cudaDeviceSynchronize());
			(getMaxResidual << <NUMBLOCKS, NUMTHREADS, NUMTHREADS * sizeof(T) >> > (initialDevGrid, glob));
			gpuErrchk(cudaDeviceSynchronize());
			(cudaMemcpy(g, glob, NUMBLOCKS*sizeof(T), cudaMemcpyDeviceToHost));
			max = 0;
			for (int i = 0; i < NUMBLOCKS; ++i) {
				if (g[i] > max) {
					max = g[i];
				}
			}
		}
		/*pressureSolve<<<NUMTHREADS, NUMBLOCKS>>>(initialDevGrid, glob);
		cudaDeviceSynchronize();*/
		(updateU << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(applyU << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(advectParticles << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(removeLists << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(resetGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(constructGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(findPureFluids << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(constructPureFluidList << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid));
		gpuErrchk(cudaDeviceSynchronize());
		(reInsertSolids << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid, states));
		gpuErrchk(cudaDeviceSynchronize());
		tElapsed += delt;
	}
	cudaDeviceSynchronize();
	delete[] g;
	delete h_max;
	cudaFree(d_max);
	cudaFree(glob);
}

template <typename T>
void CuSolver<T>::advectSingleFrameCPU() {
	grid->constructCPU();
	for (double tElapsed = 0; dt - tElapsed > 0.0001;) {
//#pragma omp parallel for
		//for (int i = 0; i < x*y*z; i++) {
		tbb::parallel_for(size_t(0), size_t(x*y*z), [&](int i){
			grid->a.a[i].aDiag = 0;
			grid->a.a[i].aX = 0;
			grid->a.a[i].aY = 0;
			grid->a.a[i].aZ = 0;
			grid->a.a[i].divU = 0;
			grid->a.a[i].p = 0;
			grid->a.a[i].u = Vec3<T>();
			grid->a.a[i].uOld = Vec3<T>();
			grid->a.numPFluids = 0;
		});
		grid->maxUCPU();
		//grid->a.dx = grid->dx;              //copy dx (not copied in Voxel copy copyFrom)
		double minT = grid->dx / (grid->a.mU + sqrt(2 * 9.8*grid->dx)), t = dt/this->grid->subSamples;
		grid->a.dt = grid->dt = minT < t ? minT : t;
		if (grid->dt + tElapsed > dt)       //correct subframe if overshooting frame duration
			grid->a.dt = grid->dt = dt - tElapsed;
		grid->a.density = grid->density;    //copy density (not copied in copyFrom)
		grid->interpUCPU();
		grid->a.interpU();
		grid->a.applyBodyForces();			//apply body force acceleration (gravity)
//#pragma omp parallel for
		//for (int i = 0; i < x*y*z; i++) {	//this loop needs to be turned into a function of CPUTriVec, calculate aDiag, aX, aY, aZ for each fluid cell
		tbb::parallel_for(size_t(0), size_t(x*y*z), [&](int i){
			if (grid->a.a[i].t == FLUID) {  //only calc divU and apply body forces on cells which are fluid
				int myz = i / (x*y), myy = (i % (x*y)) / x, myx = (i % (x*y)) % x;

				grid->a.a[i].divU = grid->a.divergenceU(myx, myy, myz);     //calculate divergence of cell

				double scale = grid->a.dt / grid->a.density / grid->a.dx / grid->a.dx;             //scale for aDiag, aX, aY, aZ, only sum aDiag if that side is nonSolid, only store aX, aY, aZ if those sides are fluid
				if (grid->a.get(myx - 1, myy, myz).t != SOLID) {
					grid->a.a[i].aDiag += scale;
				}
				if (grid->a.get(myx + 1, myy, myz).t == FLUID) {
					grid->a.a[i].aDiag += scale;
					grid->a.a[i].aX = -scale;
				}
				else if (grid->a.get(myx + 1, myy, myz).t == EMPTY) {
					grid->a.a[i].aDiag += scale;
				}

				if (grid->a.get(myx, myy - 1, myz).t != SOLID) {
					grid->a.a[i].aDiag += scale;
				}
				if (grid->a.get(myx, myy + 1, myz).t == FLUID) {
					grid->a.a[i].aDiag += scale;
					grid->a.a[i].aY = -scale;
				}
				else if (grid->a.get(myx, myy + 1, myz).t == EMPTY) {
					grid->a.a[i].aDiag += scale;
				}

				if (grid->a.get(myx, myy, myz - 1).t != SOLID) {
					grid->a.a[i].aDiag += scale;
				}
				if (grid->a.get(myx, myy, myz + 1).t == FLUID) {
					grid->a.a[i].aDiag += scale;
					grid->a.a[i].aZ = -scale;
				}
				else if (grid->a.get(myx, myy, myz + 1).t == EMPTY) {
					grid->a.a[i].aDiag += scale;
				}

				grid->a.res[i] = grid->a.a[i].divU;
			}
		});
		printf("Begin iterator\n");
		grid->a.maxResidualCPU();	                                    //single threaded search for max residual value (max divU)
		for (int i = 0; i < 1000 && grid->a.maxRes > 0.000001; i++) {   //loop, calculating new P based on parallel jacobi
			grid->a.singleThreadGS();									//Gauss Seidel for CPU
			grid->a.calcResidualCPU();									//residual calc

			grid->a.maxResidualCPU();
		}
		printf("update U\n");
		grid->a.updateUCPU();
		printf("apply U\n");
		grid->applyUCPU();                                             //interp particle U from Voxel U
		printf("advect particles\n");
		grid->advectParticlesCPU();                                   //move particles based on dt and velocity
		grid->a.resetFluidsCPU();
		grid->constructCPU();

		grid->a.findPureFluidsCPU();
		grid->reInsertCPU();

		tElapsed += grid->dt;
	}
	grid->a.dt =  grid->dt = dt;
}

#include <cuda_profiler_api.h>
template <typename T>
void CuSolver<T>::advectSingleFrameGPU() {
	cudaProfilerStart();
	cudaDeviceSynchronize();
	CuGrid<T> *initialDevGrid;
	printf("grid: %p\n", grid);
	grid->allocateOnDevice();
	cudaDeviceSynchronize();
	initialDevGrid = grid->toDeviceCopy();
	cudaDeviceSynchronize();
	cout<<"launching init\n";
	cout<<"solving\n";
	solveFrame(initialDevGrid);
	cudaDeviceSynchronize();
	cout << "preparing to copy particle array\n";
	grid->copyFromDevice(initialDevGrid);	//this calls removeDeviceCopies
	cudaDeviceSynchronize();
	cout << "cudaFree initdevgrid\n";
	cudaFree(initialDevGrid);
	cudaDeviceSynchronize();
	cudaProfilerStop();
}

template <typename T>
bool CuSolver<T>::advect() {
	CuGrid<T> *initialDevGrid;
	cudaDeviceSynchronize();
	
	grid->allocateOnDevice();
	cudaDeviceSynchronize();
	initialDevGrid = grid->toDeviceCopy();
	cudaDeviceSynchronize();
	initializeGrid<<<1, 256>>>(initialDevGrid);
	cudaDeviceSynchronize();
	copyParticles << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid);
	cudaDeviceSynchronize();
	grid->copyFromDevice(initialDevGrid);
	cudaDeviceSynchronize();
	printParts(0);
	
	for(int i = 1; i < frames; ++i){
		printf("Frame %d\n", i);
		solveFrame(initialDevGrid);
		copyParticles << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid);
		grid->copyFromDevice(initialDevGrid);
		cudaDeviceSynchronize();
		printParts(i);
	}


	clearGrid << <NUMBLOCKS, NUMTHREADS >> > (initialDevGrid);
	cudaDeviceSynchronize();
	cudaFree(initialDevGrid);

	return true;
}

template <typename T>
bool CuSolver<T>::solve() {
	if (frames <= 0 || lastSolved == -1)
		return false;
	for (int i = lastSolved; i < frames-1; i++) {
		if (!advect(i))	return false;
	}
	return true;
}

template <typename T>
void CuSolver<T>::setDimensions(double nx, double ny, double nz) {
	if (nx != dimx || ny != dimy || nz != dimz) {
		dimx = nx;
		dimy = ny;
		dimz = nz;
		x = (int)(nx / dx);
		y = (int)(ny / dx);
		z = (int)(nz / dx);
		this->grid->setDimensions(x, y, z);
	}
}

template <typename T>
void CuSolver<T>::resize(int x, int y, int z){
	this->x = x;
	this->y = y;
	this->z = z;
}

template <typename T>
void CuSolver<T>::setdx(double newX) {
	if (newX != dx && newX > 0.0001) {
		this->x = dimx / newX;
		this->y = dimy / newX;
		this->z = dimz / newX;
		this->dx = newX;
		this->grid->setdx(newX);
		this->grid->setDimensions(x, y, z);
	}
}

template <typename T>
void CuSolver<T>::setWorldPosition(double x, double y, double z) {
	worldPos = Vec3<T>(x, y, z);
}

template <typename T>
void CuSolver<T>::resizeTime(double t) {
	dt = t;
}

template <typename T>
void CuSolver<T>::printGrid(int f){
	printf("\n\nGrid Number %d:\n\n", f);
	grid->print();
}

template <typename T>
void CuSolver<T>::printParts(int f){
	grid->printParts("RadicalTest");
}

template <typename T>
void CuSolver<T>::readParticlesFromTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p) {
	//for (int i = 0; i < frames; i++)
		//particleCache[i].reserve(in->datas.Count());
	this->grid->readParticlesFromTP(in, v, p, worldPos);
}

template <typename T>
void CuSolver<T>::writeParticlesToTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p) {
	this->grid->writeParticlesToTP(in, v, p, worldPos);
}

template <typename T>
void CuSolver<T>::setSubSamples(int n) {
	this->grid->setSubSamples(n);
}

template <typename T>
void CuSolver<T>::voxelizeSolids(Triangle<T>* in, unsigned int numTris) {
#pragma omp parallel for
	for (unsigned int i = 0; i < numTris; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				
			}
		}
	}
}