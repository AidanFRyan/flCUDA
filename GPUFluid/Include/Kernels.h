#ifndef KERNELS_H
#define KERNELS_H
#include "CuGrid.h"

namespace Cu {
	//template <typename T>
	//__global__ void initializeGrid(CuGrid<T>* sourceGrid, curandState_t* states) {
	//	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	//	unsigned int offset = gridDim.x*blockDim.x;
	//	for (unsigned int i = index; i < sourceGrid->a.size; i+=offset) {
	//		sourceGrid->a.a[i].particles.setBeginEnd(&(sourceGrid->beginsEnds[i*2]), &(sourceGrid->beginsEnds[i*2 + 1]));
	//	}
	//	//if (index == 0) {
	//	//	//printf("Finished initialization\n");
	//	//}
	//}

	template<typename T>
	__global__ void copyParticles(CuGrid<T>* targetGrid) {
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("copyParticles\n");
		targetGrid->copyParticleList();
	}

	template <typename T>
	__global__ void resetGrid(CuGrid<T>* targetGrid) {
		targetGrid->construct1();
	}

	template <typename T>
	__global__ void constructGrid(CuGrid<T>* targetGrid) {	//look into sync for multi-block
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("constructing\n");
		targetGrid->construct();
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("exiting constructing\n");
	}

	template <typename T>
	__global__ void interpolateFromParts(CuGrid<T>* targetGrid) {
		/*extern __shared__ T s[];
		T *sumukx = s, *sumuky = s+ NUMTHREADS, *sumukz = s+(2 * NUMTHREADS), *sumkx = s + (3 * NUMTHREADS), *sumky = s + (4 * NUMTHREADS), *sumkz = s + (5 * NUMTHREADS);
		targetGrid->interpU(sumukx, sumuky, sumukz, sumkx, sumky, sumkz);*/
		targetGrid->interpU();
	}

	template <typename T>
	__global__ void interpolateToGrid(CuGrid<T>* targetGrid) {
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("interpToGrid\n");
		/*extern __shared__ T s[];
		T *sumukx = s, *sumuky = s + NUMTHREADS, *sumukz = s + (2 * NUMTHREADS), *sumkx = s + (3 * NUMTHREADS), *sumky = s + (4 * NUMTHREADS), *sumkz = s + (5 * NUMTHREADS), *t = s + (6 * NUMTHREADS);
		T *cPxx = s + (7 * NUMTHREADS), *cPxy = s + (8 * NUMTHREADS), *cPxz = s + (9 * NUMTHREADS), *cPyx = s + (10 * NUMTHREADS), *cPyy = s + (11 * NUMTHREADS), *cPyz = s + (12 * NUMTHREADS), *cPzx = s + (13 * NUMTHREADS), *cPzy = s + (14 * NUMTHREADS), *cPzz = s + (15 * NUMTHREADS), *dxdx = s + (16*NUMTHREADS);
		
		if (threadIdx.x == 0)
			*dxdx = targetGrid->dx * targetGrid->dx;
		__syncthreads();
		targetGrid->a.interpU(sumukx, sumuky, sumukz, sumkx, sumky, sumkz, t, cPxx, cPxy, cPxz, cPyx, cPyy, cPyz, cPzx, cPzy, cPzz, dxdx);*/
		targetGrid->a.interpU();
	}

	template <typename T>
	__global__ void copyUOld(CuGrid<T>* targetGrid) {
		int offset = gridDim.x*blockDim.x;
		for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < targetGrid->a.size; i += offset) {
			targetGrid->a.a[i].uOld = targetGrid->a.a[i].u;
		}
	}

	template <typename T>
	__global__ void getMaxU(CuGrid<T>* targetGrid, T* glob, T* max) {//look into sync for multi-block
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("maxU\n");
		targetGrid->a.maxU(glob);
		if (threadIdx.x + blockIdx.x*blockDim.x == 0)
			*max = targetGrid->a.mU;
	}

	template <typename T>
	__global__ void setParameters(CuGrid<T>* targetGrid, T delT, T tElapsed, T* max) {
		/*if (threadIdx.x + blockIdx.x*blockDim.x == 0) {
			printf("setParameters\n");
		}*/
		//	targetGrid->a.dx = targetGrid->dx;
		//}//copy dx (not copied in voxel copy copyFrom)
		double minT = targetGrid->dx / (targetGrid->a.mU + sqrt(2 * 9.8*targetGrid->dx)), t = delT / 20;
		targetGrid->a.dt = targetGrid->dt = minT < t ? minT : t;
		if (targetGrid->dt + tElapsed > delT)            				//correct subframe if overshooting frame duration
			targetGrid->a.dt = targetGrid->dt = delT - tElapsed;
		targetGrid->a.density = targetGrid->density;    				//copy density (not copied in copyFrom)
		*max = targetGrid->a.dt;
	}

	template <typename T>
	__global__ void bodyForces(CuGrid<T>* targetGrid) {
		//if (threadIdx.x + blockDim.x*blockIdx.x == 0)	printf("bodyForces\n");
		targetGrid->a.applyBodyForces();                     //apply body force acceleration (gravity)
	}

	template <typename T>
	__global__ void divergence(CuGrid<T>* targetGrid) {
		extern __shared__ T aD[];
		int offset = blockDim.x*gridDim.x;
		for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < targetGrid->a.size; i += offset) {  //this loop needs to be turned into a function of TriVec, calculate aDiag, aX, aY, aZ for each fluid cell                                                                //Zero out all coefficients, all velocities, all pressures, and divergences
			aD[threadIdx.x] = 0;
			if (targetGrid->a.a[i].t == FLUID) {                                          //only calc divU and apply body forces on cells which are fluid
				int myz = i / (targetGrid->x*targetGrid->y), myy = (i % (targetGrid->x*targetGrid->y)) / targetGrid->x, myx = (i % (targetGrid->x*targetGrid->y)) % targetGrid->x;
				targetGrid->a.a[i].divU = targetGrid->a.divergenceU(myx, myy, myz);     //calculate divergence of cell
				double scale = targetGrid->a.dt / targetGrid->a.density / targetGrid->a.dx / targetGrid->a.dx;             //scale for aDiag, aX, aY, aZ, only sum aDiag if that side is nonSolid, only store aX, aY, aZ if those sides are fluid
				if (targetGrid->a.get(myx - 1, myy, myz).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].anX = -scale;
				}
				else if (targetGrid->a.get(myx - 1, myy, myz).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}
				if (targetGrid->a.get(myx + 1, myy, myz).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].aX = -scale;
				}
				else if (targetGrid->a.get(myx + 1, myy, myz).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}

				if (targetGrid->a.get(myx, myy - 1, myz).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].anY = -scale;
				}
				else if (targetGrid->a.get(myx, myy - 1, myz).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}
				if (targetGrid->a.get(myx, myy + 1, myz).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].aY = -scale;
				}
				else if (targetGrid->a.get(myx, myy + 1, myz).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}

				if (targetGrid->a.get(myx, myy, myz - 1).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].anZ = -scale;
				}
				else if (targetGrid->a.get(myx, myy, myz - 1).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}
				if (targetGrid->a.get(myx, myy, myz + 1).t == FLUID) {
					aD[threadIdx.x] += scale;
					targetGrid->a.a[i].aZ = -scale;
				}
				else if (targetGrid->a.get(myx, myy, myz + 1).t == EMPTY) {
					aD[threadIdx.x] += scale;
				}

				targetGrid->a.a[i].aDiag = aD[threadIdx.x];
				targetGrid->a.res[i] = targetGrid->a.a[i].divU;
			}
		}
	}

	template <typename T>
	__global__ void pressureSolve(CuGrid<T>* targetGrid, T* glob) {	//praying this synchronizes enough that the solve works
		targetGrid->a.maxResidual(glob);
		__syncthreads();
		for (int j = 0; j < 3000 && targetGrid->a.maxRes > 0.000001; ++j) {
			targetGrid->a.multiThreadJacobi();
			__syncthreads();
			targetGrid->a.calcResidualGS();
			__syncthreads();
			targetGrid->a.maxResidual(glob);
			__syncthreads();
		}
	}

	template <typename T>
	__global__ void getMaxResidual(CuGrid<T>* targetGrid, T* glob) {//look into sync for multi-block usage
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("getMaxResidual\n");
		extern __shared__ T s[];
		targetGrid->a.maxResidual(s, glob);
	}

	template <typename T>
	__global__ void jacobi(CuGrid<T>* targetGrid) {
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("jacobi\n");
		extern __shared__ T s[];
		targetGrid->a.multiThreadJacobi(s);
		targetGrid->a.calcResidualGS();
	}

	template <typename T>
	__global__ void redSOR(CuGrid<T>* targetGrid) {
		targetGrid->a.redSOR();
	}

	template <typename T>
	__global__ void blackSOR(CuGrid<T>* targetGrid) {
		targetGrid->a.blackSOR();
	}

	template <typename T>
	__global__ void asyncSOR(CuGrid<T>* targetGrid) {
		targetGrid->a.redSOR();
		targetGrid->a.blackSOR();
		targetGrid->a.calcResidualGS();
	}

	template <typename T>
	__global__ void residual(CuGrid<T>* targetGrid) {
		//if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("calc residual\n");
		
		targetGrid->a.calcResidualGS();
	}

	template <typename T>
	__global__ void updateU(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("updateU\n");
		targetGrid->a.updateU();
	}

	template <typename T>
	__global__ void applyU(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("applyU\n");
		/*extern __shared__ T s[];
		T *ux = s, *uy = s + NUMTHREADS, *uz = s + (NUMTHREADS * 2), *uOldx = s+(NUMTHREADS *3), *uOldy = s + (NUMTHREADS * 4), *uOldz = s + (NUMTHREADS * 5);
		for (int i = threadIdx.x; i < targetGrid->a.size; i += blockDim.x) {
			Vec3<T> tu = targetGrid->a.a[i].u, tuOld = targetGrid->a.a[i].uOld;
			ux[i] = tu.x;
			uy[i] = tu.y;
			uz[i] = tu.z;

			uOldx[i] = tuOld.x;
			uOldy[i] = tuOld.y;
			uOldz[i] = tuOld.z;
		}
		__syncthreads();*/
		targetGrid->applyU();
	}

	template <typename T>
	__global__ void advectParticles(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("advectParticles\n");
		targetGrid->advectParticles();
	}

	template <typename T>
	__global__ void reInsertSolids(CuGrid<T>* targetGrid, curandState_t* states) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("reInsertSolids\n");
		targetGrid->reInsert(states);
	}

	template <typename T>
	__global__ void removeLists(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("removeLists\n");
		targetGrid->a.removeAandLists();
	}

	__global__ void initStates(curandState_t* in) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("initStates\n");
		int offset = gridDim.x*blockDim.x;
		for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < NUMBLOCKS*NUMTHREADS; i += offset) {
			curand_init(i, i, 0, in + i);
		}
	}

	template <typename T>
	__global__ void findPureFluids(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("findPureFluids\n");
		targetGrid->a.findPureFluids();
	}

	template <typename T>
	__global__ void constructPureFluidList(CuGrid<T>* targetGrid) {
		if (threadIdx.x + blockIdx.x*blockDim.x == 0) printf("constructPureFluidList\n");
		targetGrid->a.constructPureFluidList();
	}

	template <typename T>
	__global__ void solveForFrame(CuGrid<T>* targetGrid, CuGrid<T>** gridUpdate, int frames, T* glob) {
		int x = targetGrid->x;
		int y = targetGrid->y;
		int z = targetGrid->z;
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		int offset = gridDim.x*blockDim.x;
		int maxGridIndex = x * y*z;
		double delT = targetGrid->dt;
		__syncthreads();

		targetGrid->copyParticleList();
		__syncthreads();
		if(index == 0) printf("finished copying\n");
		__syncthreads();
		while (*gridUpdate != 0) { __syncthreads(); }	                            //hold while cpu copies
		__syncthreads();
		if (index == 0) {
			*gridUpdate = targetGrid;
		}
		__syncthreads();
		for (int curFrame = 1; curFrame < frames; ++curFrame) {       //for each frame
			if (index == 0)	printf("Frame %d\n", curFrame);
			for (double tElapsed = 0; delT - tElapsed > 0.0001;) {    //loop subframes until we reach dt of the frame
				if (index == 0) printf("Constructing\n");
				targetGrid->a.construct(targetGrid->list, targetGrid->numParticles);

				__syncthreads();
				if (index == 0) printf("Interpolating to Grid\n");
				__syncthreads();
				targetGrid->a.interpU();                            //interpolate from particle velocity to grid
				__syncthreads();
				if (index == 0) printf("Finding MaxU\n");
				targetGrid->a.maxU(glob);
				__syncthreads();
				if (index == 0) {
					targetGrid->a.dx = targetGrid->dx;              				//copy dx (not copied in voxel copy copyFrom)
					double minT = targetGrid->dx / (targetGrid->a.mU + sqrt(2 * 9.8*targetGrid->dx)), t = delT / 20.0;
					targetGrid->a.dt = targetGrid->dt = minT < t ? minT : t;
					if (targetGrid->dt + tElapsed > delT)            				//correct subframe if overshooting frame duration
						targetGrid->a.dt = targetGrid->dt = delT - tElapsed;
					targetGrid->a.density = targetGrid->density;    				//copy density (not copied in copyFrom)
				}
				__syncthreads();
				if (index == 0) printf("Applying Body Forces\n");
				for (int i = index; i < maxGridIndex; i += offset) {  //this loop needs to be turned into a function of TriVec, calculate aDiag, aX, aY, aZ for each fluid cell
					if (targetGrid->a.a[i].t == FLUID) {                                          //only calc divU and apply body forces on cells which are fluid
						targetGrid->a.a[i].applyBodyForces(targetGrid->dt);                     //apply body force acceleration (gravity)
					}
				}
				__syncthreads();
				if (index == 0) printf("Divergence/Coefficient\n");
				for (int i = index; i < maxGridIndex; i += offset) {  //this loop needs to be turned into a function of TriVec, calculate aDiag, aX, aY, aZ for each fluid cell                                                                //Zero out all coefficients, all velocities, all pressures, and divergences
					if (targetGrid->a.a[i].t == FLUID) {                                          //only calc divU and apply body forces on cells which are fluid
						int myz = i / (x*y), myy = (i % (x*y)) / x, myx = (i % (x*y)) % x;

						targetGrid->a.a[i].divU = targetGrid->a.divergenceU(myx, myy, myz);     //calculate divergence of cell
						double scale = targetGrid->a.dt / targetGrid->a.density / targetGrid->a.dx / targetGrid->a.dx;             //scale for aDiag, aX, aY, aZ, only sum aDiag if that side is nonSolid, only store aX, aY, aZ if those sides are fluid

						if (targetGrid->a.get(myx - 1, myy, myz).t != SOLID) {
							targetGrid->a.a[i].aDiag += scale;
						}
						if (targetGrid->a.get(myx + 1, myy, myz).t == FLUID) {
							targetGrid->a.a[i].aDiag += scale;
							targetGrid->a.a[i].aX = -scale;
						}
						else if (targetGrid->a.get(myx + 1, myy, myz).t == EMPTY) {
							targetGrid->a.a[i].aDiag += scale;
						}

						if (targetGrid->a.get(myx, myy - 1, myz).t != SOLID) {
							targetGrid->a.a[i].aDiag += scale;
						}
						if (targetGrid->a.get(myx, myy + 1, myz).t == FLUID) {
							targetGrid->a.a[i].aDiag += scale;
							targetGrid->a.a[i].aY = -scale;
						}
						else if (targetGrid->a.get(myx, myy + 1, myz).t == EMPTY) {
							targetGrid->a.a[i].aDiag += scale;
						}

						if (targetGrid->a.get(myx, myy, myz - 1).t != SOLID) {
							targetGrid->a.a[i].aDiag += scale;
						}
						if (targetGrid->a.get(myx, myy, myz + 1).t == FLUID) {
							targetGrid->a.a[i].aDiag += scale;
							targetGrid->a.a[i].aZ = -scale;
						}
						else if (targetGrid->a.get(myx, myy, myz + 1).t == EMPTY) {
							targetGrid->a.a[i].aDiag += scale;
						}

						targetGrid->a.a[i].res = targetGrid->a.a[i].divU;
					}
				}
				__syncthreads();
				if (index == 0) printf("Finding Max Residual\n");
				targetGrid->a.maxResidual(glob);                                        //single threaded search for max residual value (max divU)
				__syncthreads();
				if (index == 0)	printf("Pressure Solve\n");
				for (int i = 0; i < 3000 && targetGrid->a.maxRes > 0.000001; i++) {    //loop, calculating new P based on parallel jacobi

					targetGrid->a.multiThreadJacobi();       	                    //parallel jacobi
					targetGrid->a.calcResidualGS();                                //parallel residual calc
					__syncthreads();
					targetGrid->a.maxResidual(glob);                                    //single threaded residual search
					__syncthreads();
					if (index == 0)
						if (i == 2999)
							printf("Warning: max iterations reached. Residual: %f\n", targetGrid->a.maxRes);
					__syncthreads();
					//g.sync();
				}
				__syncthreads();
				//g.sync();
				if (index == 0) printf("update U\n");
				targetGrid->a.updateU();                                            //update voxel U with pressure term
				//g.sync();
				__syncthreads();
				if (index == 0) printf("apply U\n");
				targetGrid->a.applyU();                                             //interp particle U from voxel U
				//g.sync();
				__syncthreads();
				if (index == 0) printf("advect particles\n");
				targetGrid->a.advectParticles();                                   //move particles based on dt and velocity
				tElapsed += targetGrid->dt;                                         //add dt to total time elapsed
				targetGrid->a.removeAandLists();
				__syncthreads();
				//g.sync();
			}
			// if(index == 0) printf("copying to source\n");
			// sourceGrid->a.copyFrom(targetGrid->a);                                  //copy from target to source TriVec
			targetGrid->copyParticleList();
			__syncthreads();
			//g.sync();
			// g.sync();
			if(index == 0) printf("finished copying\n");
			__syncthreads();
			//g.sync();
			while (*gridUpdate != 0) { __syncthreads(); }	                            //hold while cpu copies
			printf("escaped\n");
			__syncthreads();
			//g.sync();
			if (index == 0) {
				*gridUpdate = targetGrid;
			}
			__syncthreads();
			//g.sync();
		}
	}

}
#endif