#include "CPUSolver.h"
#define NUMBLOCKS 1
#define NUMTHREADS 1
using namespace Cu;


template <typename T>
CPUSolver<T>::CPUSolver(int x, int y, int z, int frames, double dx, double t) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->frames = frames;
	this->dt = t;
	lastSolved = -1;
	sourceGrid = new CPUGrid<T>(x, y, z, dx, t, 997.0);
	targetGrid = new CPUGrid<T>(x, y, z, dx, t, 997.0);
}

template <typename T>
CPUSolver<T>::~CPUSolver() {
	for(int i = 0; i < frames; i++){
		if(sourceGrid != nullptr){
			delete sourceGrid;
			sourceGrid = nullptr;
		}
		if(targetGrid != nullptr){
			delete targetGrid;
			targetGrid = nullptr;
		}
	}
}

template <typename T>
void CPUSolver<T>::testInit(){
	for(int k = 0; k < z; k++){
		for(int j = 0; j < y; j++){
			for(int i = 0; i < x; i++){
				CPUVoxel<T>* t = &sourceGrid->a.get(i, j, k);
				// if(i == 0 || i == x-1 || j == 0 || j == y-1 || k == 0 || k == z-1){
				if (i < 5 * x / 8 && i > 3 * x / 8 && k < 5 * z / 8 && k > 3*z/8){// && j < 5 * y / 8 && j > 3 * y / 8){
					t->t = SOLID;
				}
				// // else if(i < 5*x/8 && i > 3*x/8 && j < 5*y/8 && j > 3*y/8)
				else
				if(j < 5*y/8 && j > 3*y/8)
					t->t = FLUID;
				else{
					t->t = EMPTY;
				}
			}
		}
	}
}

template <typename T>
bool CPUSolver<T>::initialValues(CPUGrid<T>* initial) {
	if (!frames ||	initial->x*initial->y*initial->z != x * y*z) return false;
	sourceGrid = initial;
	lastSolved = 0;
	return true;
}

template <typename T>
bool CPUSolver<T>::advect() {
	targetGrid->a.copyFrom(sourceGrid->a);

	targetGrid->initializeParticles();
	this->printParts(0);
	double delT = sourceGrid->dt;

	for(int c = 1; c < frames; c++){

		printf("Frame %d\n", c);
		for(double tElapsed = 0; delT - tElapsed > 0.0001;){
#pragma omp parallel for
			for(int i = 0; i < x*y*z; i++){
				targetGrid->a.a[i].aDiag = 0;
				targetGrid->a.a[i].aX = 0;
				targetGrid->a.a[i].aY = 0;
				targetGrid->a.a[i].aZ = 0;
				targetGrid->a.a[i].divU = 0;
                targetGrid->a.a[i].p = 0;
                targetGrid->a.a[i].u = Vec3<T>();
                targetGrid->a.a[i].uOld = Vec3<T>();
				targetGrid->a.pureFluidList.clear();
			}

			targetGrid->maxU();
			targetGrid->a.dx = sourceGrid->dx;              //copy dx (not copied in CPUVoxel copy copyFrom)
			// targetGrid->a.dt = targetGrid->dt = sourceGrid->dt = sourceGrid->dx / (sourceGrid->a.mU + sqrt(2*9.8*targetGrid->dx)); //set dt for subframe
			targetGrid->a.dt = targetGrid->dt = sourceGrid->dt = delT / 4.0;
			if (targetGrid->dt + tElapsed > delT)            //correct subframe if overshooting frame duration
				targetGrid->a.dt = targetGrid->dt = sourceGrid->dt = delT - tElapsed;
			targetGrid->a.density = sourceGrid->density;    //copy density (not copied in copyFrom)
			targetGrid->a.interpU();
			
#pragma omp parallel for
			for(int i = 0; i < x*y*z; i++){  //this loop needs to be turned into a function of CPUTriVec, calculate aDiag, aX, aY, aZ for each fluid cell
				if(targetGrid->a.a[i].t == FLUID){                                          //only calc divU and apply body forces on cells which are fluid
					int myz = i/(x*y), myy = (i%(x*y))/x, myx = (i%(x*y))%x;
					targetGrid->a.a[i].applyBodyForces(targetGrid->dt);                     //apply body force acceleration (gravity)
					targetGrid->a.a[i].divU = targetGrid->a.divergenceU(myx, myy, myz);     //calculate divergence of cell

					double scale = targetGrid->a.dt/targetGrid->a.density/targetGrid->a.dx/targetGrid->a.dx;             //scale for aDiag, aX, aY, aZ, only sum aDiag if that side is nonSolid, only store aX, aY, aZ if those sides are fluid
					
					
					if(targetGrid->a.get(myx-1, myy, myz).t != SOLID){
						targetGrid->a.a[i].aDiag += scale;
					}
					if(targetGrid->a.get(myx+1, myy, myz).t == FLUID){
						targetGrid->a.a[i].aDiag += scale;
						targetGrid->a.a[i].aX = -scale;
					}
					else if(targetGrid->a.get(myx+1, myy, myz).t == EMPTY){
						targetGrid->a.a[i].aDiag += scale;
					}
					
					if(targetGrid->a.get(myx, myy-1, myz).t != SOLID){
						targetGrid->a.a[i].aDiag += scale;
					}
					if(targetGrid->a.get(myx, myy+1, myz).t == FLUID){
						targetGrid->a.a[i].aDiag += scale;
						targetGrid->a.a[i].aY = -scale;
					}
					else if(targetGrid->a.get(myx, myy+1, myz).t == EMPTY){
						targetGrid->a.a[i].aDiag += scale;
					}

					if(targetGrid->a.get(myx, myy, myz-1).t != SOLID){
						targetGrid->a.a[i].aDiag += scale;
					}
					if(targetGrid->a.get(myx, myy, myz+1).t == FLUID){
						targetGrid->a.a[i].aDiag += scale;
						targetGrid->a.a[i].aZ = -scale;
					}
					else if(targetGrid->a.get(myx, myy, myz+1).t == EMPTY){
						targetGrid->a.a[i].aDiag += scale;
					}

					targetGrid->a.a[i].res = targetGrid->a.a[i].divU;
				}
			}
			printf("Begin iterator\n");
			targetGrid->a.maxResidual();	                                      //single threaded search for max residual value (max divU)
			double prevRes = 0.0;
			for(int i = 0; i < 1000 && targetGrid->a.maxRes > 0.000001; i++){    //loop, calculating new P based on parallel jacobi
                //printf("Iteration %d %f\n", i, targetGrid->a.maxRes);
				targetGrid->a.singleThreadGS();                             //Gauss Seidel for CPU
				targetGrid->a.calcResidualGS();                                //residual calc

				targetGrid->a.maxResidual();      //single threaded residual search
				/*if(i == 999)
					printf("Warning: max iterations reached. Residual: %f\n", targetGrid->a.maxRes);*/
			}
			printf("update U\n");
			targetGrid->a.updateU();
			printf("apply U\n");
			targetGrid->applyU();                                             //interp particle U from CPUVoxel U
			printf("advect particles\n");
			targetGrid->advectParticles();                                   //move particles based on dt and velocity
			targetGrid->a.resetFluids();
			targetGrid->setFluidCells();

			targetGrid->a.findPureFluids();
			targetGrid->reinsertOOBParticles();

			tElapsed += targetGrid->dt;       
		}
		sourceGrid->a.copyFrom(targetGrid->a);

		this->printParts(c);
	}

	return true;
}

template <typename T>
bool CPUSolver<T>::solve() {
	if (frames <= 0 || lastSolved == -1)
		return false;
	for (int i = lastSolved; i < frames-1; i++) {
		if (!advect(i))	return false;
	}
	return true;
}

template <typename T>
void CPUSolver<T>::resize(int x, int y, int z){
	this->x = x;
	this->y = y;
	this->z = z;
}

template <typename T>
void CPUSolver<T>::resizeTime(int t) {
	frames = t;
}

template <typename T>
void CPUSolver<T>::printGrid(int f){
	printf("\n\nGrid Number %d:\n\n", f);
	sourceGrid->print();
}

template <typename T>
void CPUSolver<T>::printParts(int f){
	targetGrid->printParts(to_string(f));
}
