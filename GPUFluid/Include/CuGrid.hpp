#include "CuGrid.h"

template <typename T>
CuGrid<T>::CuGrid() {
	x = 0;
	y = 0;
	z = 0;
	this->dt = 0;
	this->dx = 0;
	this->density = 0;
	this->subSamples = 4;
	this->list = nullptr;
	this->d_list = nullptr;
	this->numParticles = 0;
}

template <typename T>
CuGrid<T>::CuGrid(unsigned int x, unsigned int y, unsigned int z, double dx, double dt, double density) : CuGrid() {
	this->x = x;
	this->y = y;
	this->z = z;
	printf("allocing %lu bytes\n", sizeof(Voxel<T>)*x*y*z);
	this->a = std::move(TriVec<T>(x, y, z, dx, dt, density));

	this->d_a = std::move(TriVec<T>(x, y, z, dx, dt, density, true));

	this->dx = dx;
	this->dt = dt;
	this->density = density;
}



template <typename T>
void CuGrid<T>::setTstep(double t){
	this->dt = t;
	this->a.dt = t;
}

template <typename T>
void CuGrid<T>::printParts(string filename){
	ofstream f;
	f.open(filename);
	for(int i = 0; i < numParticles; i++){
		f<<list[i].p.x<<' '<<list[i].p.y<<' '<<list[i].p.z<<endl;
	}
	f.close();
}

template <typename T>
CuGrid<T>::CuGrid(const CuGrid<T>& in) : CuGrid() {	//there was a reason I used the move version of the TriVec assignment operator... probably to do with the
													//copy of pointers and preventing things from going out of scope
	this->x = in.x;
	this->y = in.y;
	this->z = in.z;
	this->dx = in.dx;
	this->dt = in.dt;
	this->density = in.density;
	this->numParticles = in.numParticles;
	this->a = std::move(TriVec<T>(x, y, z, dx, dt, density));
	this->subSamples = in.subSamples;
	//this->particleLists = in.particleLists;
	this->d_a = std::move(TriVec<T>(x, y, z, dx, dt, density, true));
	//this->beginsEnds = in.beginsEnds;
	
}

template <typename T>
const CuGrid<T>& CuGrid<T>::operator=(const CuGrid<T>& in){
	a = in.a;
	d_a = in.d_a;
	x = in.x;	
	y = in.y;
	z = in.z;
	dx = in.dx;
	dt = in.dt;
	density = in.density;
	numParticles = in.numParticles;
	list = in.list;
	//copyList = in.copyList;
	d_list = in.d_list;
	subSamples = in.subSamples;
	//particleLists = in.particleLists;
	//beginsEnds = in.beginsEnds;
	return *this;
}

template <typename T>
void CuGrid<T>::removeDeviceTriVec(){
	cudaFree(d_a.a);
	//cudaFree(particleLists);
	cudaFree(d_a.res);
	cudaFree(d_a.ux);
	cudaFree(d_a.uy);
	cudaFree(d_a.uz);
	cudaFree(beginsEnds);
}

template <typename T>
CuGrid<T>::~CuGrid() {	//memory leak with list
	//removeDeviceCopies();

}

template <typename T>
void CuGrid<T>::removeDeviceCopies() {
	printf("deleting %p\n", this);
	if (d_a.a != nullptr) {
		cudaFree(d_a.a);
		d_a.a = nullptr;
	}
	if (d_a.res != nullptr) {
		cudaFree(d_a.res);
		d_a.res = nullptr;
	}
	if (d_a.pFluidIndexes != nullptr) {
		cudaFree(d_a.pFluidIndexes);
		d_a.pFluidIndexes = nullptr;
	}
}

template <typename T>
void CuGrid<T>::nullifyHostTriVecs() {
	a.a = nullptr;
	a.res = nullptr;
	a.pFluidIndexes = nullptr;
}

template <typename T>
void CuGrid<T>::nullifyDeviceTriVecs() {
	d_a.a = nullptr;
	d_a.res = nullptr;
	d_a.pFluidIndexes = nullptr;
}

template <typename T>
bool CuGrid<T>::allocateOnDevice() {
	cudaDeviceSynchronize();
	int size = x * y*z;
	cudaMalloc((void**)&d_a.a, sizeof(Voxel<T>)*size);
	cudaMalloc((void**)&d_a.res, sizeof(T)*size);

	cudaMalloc((void**)&d_a.pFluidIndexes, sizeof(int)*size);

	return true;
}

template <typename T>
CuGrid<T>* CuGrid<T>::toDeviceCopy() {
	CuGrid<T> *d_temp, temp;
	temp = *this;

	cudaMemcpy(d_a.a, a.a, sizeof(Voxel<T>)*x*y*z, cudaMemcpyHostToDevice);

	temp.list = d_list;
	temp.a = d_a;
	cudaMalloc((void**)&d_temp, sizeof(CuGrid<T>));	//need to cudaFree the return value
	cudaDeviceSynchronize();
	cudaMemcpy(d_temp, &temp, sizeof(CuGrid<T>), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	temp.a.a = nullptr;	//null these out so they aren't deleted
	temp.d_a.a = nullptr;
	temp.d_a.res = nullptr;
	temp.a.res = nullptr;
	return d_temp;
}

template <typename T>
void CuGrid<T>::initializeList(int n){
	cudaMalloc((void**)&d_list, sizeof(Particle<T>)*n);
	cudaDeviceSynchronize();
	list = new Particle<T>[n];
	cudaMemcpy(d_list, this->list, sizeof(Particle<T>)*n, cudaMemcpyHostToDevice);
	numParticles = n;
	cudaDeviceSynchronize();
}

template <typename T>
bool CuGrid<T>::copyFromDevice(CuGrid<T>* d_grid) {
	//cout << "copying from device, creating temp grid/n";
	CuGrid<T>* grid = new CuGrid<T>;
	cudaMemcpy(grid, d_grid, sizeof(CuGrid<T>), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if(grid->x != x || grid->y != y || grid->z != z || grid->numParticles != numParticles){
		//cout << "error, d_grid dimensions don't match\n";
		delete grid;
		return false;
	}
	//cout << "copying list of particles\n";
	cudaMemcpy(this->list, grid->d_list, sizeof(Particle<T>)*grid->numParticles, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	grid->d_a.res = nullptr;
	grid->a.res = nullptr;
	grid->a.a = nullptr;
	grid->d_a.a = nullptr;
	delete grid;
	removeDeviceCopies();
	return true;
}

template <typename T>
bool CuGrid<T>::copyFromDeviceAsync(CuGrid<T>* d_grid, cudaStream_t& stream) {
	CuGrid<T>* grid = new CuGrid<T>;
	cudaMemcpyAsync(grid, d_grid, sizeof(CuGrid<T>), cudaMemcpyDeviceToHost, stream);

	if(grid->x != x || grid->y != y || grid->z != z){
		delete grid;
		return false;
	}

	cudaMemcpyAsync(this->list, grid->copyList, sizeof(Particle<T>)*grid->numParticles, cudaMemcpyDeviceToHost, stream);
	grid->nullifyHostTriVecs();

	delete grid;

	return true;
}



template <typename T>
CuGrid<T>* CuGrid<T>::toDeviceEmpty() {
	CuGrid<T> temp, *d_temp;
	temp.a = d_a;
	temp.x = x;
	temp.y = y;
	temp.z = z;

	cudaMalloc((void**)&d_temp, sizeof(CuGrid<T>));
	cudaMemcpy(d_temp, &temp, sizeof(CuGrid<T>), cudaMemcpyHostToDevice);

	temp.nullifyHostTriVecs();

	return d_temp;
}

template <typename T>
void CuGrid<T>::setDimensions(int x, int y, int z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->a.setDimensions(x, y, z);
	this->d_a.setDimensions(x, y, z);
}

template <typename T>
void CuGrid<T>::setdx(double nx) {
	this->dx = nx;
	this->a.setdx(nx);
	this->d_a.setdx(nx);
}

template <typename T>
void CuGrid<T>::print(){
	for(int k = z-1; k >= 0; k--){
		printf("\nZ = %d\n", k);
		for(int j = y-1; j >= 0; j--){
			for(int i = x-1; i >= 0; i--){
				Voxel<T> *t = &a.get(i, j, k);
				printf("| %f |", t->p);
			}
			printf("\n");
		}
	}
}

template <typename T>
void CuGrid<T>::readParticlesFromTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p, const Vec3<T>& offsetP) {	//currently set up so it doesn't know it can keep a copy on GPU at all times
	//printf("v %p p %p\n", v, p);
	if (in->datas.Count() != numParticles) {
		if (this->list != nullptr) {
			delete[] this->list;
			this->list = nullptr;
		}
		if (this->d_list != nullptr) {
			cudaFree(this->d_list);
			this->d_list = nullptr;
		}
	}
	if (this->list == nullptr) {
		this->list = new Particle<T>[in->datas.Count()];
	}
	if (this->d_list == nullptr) {
		cudaMalloc((void**)&d_list, in->datas.Count() * sizeof(Particle<T>));
	}
	if (p == nullptr || v == nullptr) {
		//for (int i = 0; i < in->datas.Count(); ++i) {
		tbb::parallel_for(size_t(0), size_t(in->datas.Count()), [&](int i) {
			this->list[i].p = in->datas[i].pos;
			this->list[i].p = this->list[i].p - offsetP;
			this->list[i].v = in->datas[i].vel;
		}
		);
	}
	else if (v != nullptr && p != nullptr) {
//#pragma omp parallel for
		//for (int i = 0; i < in->datas.Count(); ++i) {
		tbb::parallel_for(size_t(0), size_t(in->datas.Count()), [&](int i){
			this->list[i].p = p[i];
			this->list[i].v = v[i];
		});
		delete[] v;	//inefficient but a simple way to handle changes in particle counts between frames
		delete[] p;
		v = nullptr;
		p = nullptr;
	}
	cudaDeviceSynchronize();
	cudaMemcpy(d_list, this->list, in->datas.Count() * sizeof(Particle<T>), cudaMemcpyHostToDevice);
	this->numParticles = in->datas.Count();
}

template <typename T>
void CuGrid<T>::writeParticlesToTP(FlipFluidBasisThreadData* in, Vec3<T>*& v, Vec3<T>*& p, const Vec3<T>& offsetP) {	//only copy average velocity from frame, store velocity/positions in separate arrays.
	if (v == nullptr) {
		v = new Vec3<T>[in->datas.Count()];
	}
	if (p == nullptr) {
		p = new Vec3<T>[in->datas.Count()];
	}
	//cudaMemcpy(list, d_list, sizeof(Particle<T>)*in->datas.Count(), cudaMemcpyDeviceToHost);
//#pragma omp parallel for
	//for (int i = 0; i < in->datas.Count(); ++i) {
	tbb::parallel_for(size_t(0), size_t(in->datas.Count()), [&](int i){
		v[i] = this->list[i].v;
		p[i] = this->list[i].p;

		in->datas[i].vel.x = (this->list[i].p.x + offsetP.x - in->datas[i].pos.x) / in->dt;	//there is loss of precision here since in->datas stores positions as floats!
		in->datas[i].vel.y = (this->list[i].p.y + offsetP.y - in->datas[i].pos.y) / in->dt;
		in->datas[i].vel.z = (this->list[i].p.z + offsetP.z - in->datas[i].pos.z) / in->dt;
	});
}

template <typename T>
__device__ void CuGrid<T>::advectParticles() {													//advect particles!
	int offset = blockDim.x*gridDim.x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < numParticles; i += offset) {
		Particle<T> k2 = this->list[i], k3;
		k2.p += k2.v*(dt / 2.0);
		a.interpUtoP(k2);
		k3 = k2;
		k3.p += k3.v*dt*(3.0 / 4.0);
		a.interpUtoP(k3);
		this->list[i].p += this->list[i].v*dt*(2.0 / 9.0) + k2.v*dt*(3.0 / 9.0) + k3.v*dt*(4.0 / 9.0);
	}
}

template <typename T>
void CuGrid<T>::advectParticlesCPU() {
//#pragma omp parallel for
	//for (int p = 0; p < numParticles; ++p) {
	tbb::parallel_for(size_t(0), size_t(numParticles), [&](int p){
		Particle<T> k2 = this->list[p], k3;
		k2.p += k2.v*(dt / 2.0);
		a.interpUtoP(k2);
		k3 = k2;
		k3.p += k3.v*dt*(3.0 / 4.0);
		a.interpUtoP(k3);
		this->list[p].p += dt*(this->list[p].v*(2.0 / 9.0) + k2.v*(3.0 / 9.0) + k3.v*(4.0 / 9.0));
	});
}

template <typename T>
__device__ void CuGrid<T>::reInsert(curandState_t* states) {
	int offset = gridDim.x*blockDim.x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < numParticles; i += offset) {
		if (a.get(floor(this->list[i].p.x / dx), floor(this->list[i].p.y / dx), floor(this->list[i].p.z / dx)).t == SOLID) {
			a.insertToRandomPureFluid(this->list[i], states);
		}
	}
}

template <typename T>
void CuGrid<T>::reInsertCPU() {
//#pragma omp parallel for
	//for (int p = 0; p < numParticles; ++p) {
	tbb::parallel_for(size_t(0), size_t(numParticles), [&](int p){
		int i = floor(this->list[p].p.x / dx), j = floor(this->list[p].p.y / dx), k = floor(this->list[p].p.z / dx);
		if (a.get(i, j, k).t == SOLID) {
			int i = a.pFluidIndexes[(int)(a.dist->operator()(*(a.gen))*a.numPFluids)];
			int tz = i / (x*y), ty = (i % (x*y)) / x, tx = (i % (x*y)) % x;
			this->list[p].p.x = tx * dx + a.dist->operator()(*(a.gen))*dx;
			this->list[p].p.y = ty * dx + a.dist->operator()(*(a.gen))*dx;
			this->list[p].p.z = tz * dx + a.dist->operator()(*(a.gen))*dx;
			a.interpUtoP(this->list[p]);
		}
	});
}

template <typename T>
__device__ void CuGrid<T>::construct1() {
	int offset = gridDim.x*blockDim.x, size = x * y*z;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < size; i += offset) {
		if (a.a[i].t != SOLID) {
			a.a[i].t = EMPTY;
			a.a[i].sd = 1;
		}
	}
}

template <typename T>
__device__ void CuGrid<T>::construct() {
	int offset = gridDim.x*blockDim.x;
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < numParticles; i += offset) {
		int tx = floor(this->list[i].p.x / dx), ty = floor(this->list[i].p.y / dx), tz = floor(this->list[i].p.z / dx);
		//if (a.get(tx, ty, tz).t == EMPTY) {
			a.get(tx, ty, tz).t = FLUID;
			a.a[i].sd = -1;
		//}
	}
}

template <typename T>
void CuGrid<T>::constructCPU() {
//#pragma omp parallel for
//	for (int p = 0; p < numParticles; ++p) {
	tbb::parallel_for(size_t(0), size_t(x*y*z), [&](int i) {
		if (a.a[i].t != SOLID) {
			a.a[i].t = EMPTY;
			a.a[i].sd = 1;
		}
	});
	tbb::parallel_for(size_t(0), size_t(numParticles), [&](int p){
		int i = floor(this->list[p].p.x / dx), j = floor(this->list[p].p.y / dx), k = floor(this->list[p].p.z / dx);
		//if (a.get(i, j, k).t == EMPTY) {
			a.get(i, j, k).t = FLUID;
			a.a[i].sd = -1;
		//}
	});

	
}

template <typename T>
__device__ void CuGrid<T>::applyU() {											//apply grid U to particles
	int offset = gridDim.x*blockDim.x;											//for each fluid voxel, run interpUtoP for each particle inside
	for (int l = threadIdx.x + blockDim.x*blockIdx.x; l < numParticles; l += offset) {
		a.interpUtoP(this->list[l]);
	}
}

template <typename T>
void CuGrid<T>::applyUCPU() {
//#pragma omp parallel for
//	for (int p = 0; p < numParticles; p++) {
	tbb::parallel_for(size_t(0), size_t(numParticles), [&](int p){
		a.interpUtoP(this->list[p]);
	});
}

template <typename T>
__device__ void CuGrid<T>::interpU(T* sumukx, T* sumuky, T* sumukz, T* sumkx, T* sumky, T* sumkz) {
	int o = gridDim.x*blockDim.x;
	T offset = dx / 2;
	for (int l = threadIdx.x + blockIdx.x*blockDim.x; l < numParticles; l += o) {
		Particle<T> p = this->list[l];
		int tx = floor(p.p.x / dx), ty = floor(p.p.y / dx), tz = floor(p.p.z / dx);
		for (int i = -2; i < 3; ++i) {
			for (int j = -2; j < 3; ++j) {
				for (int k = -2; k < 3; ++k) {
					if (a.get(tx + i, ty + j, tz + k).t != SOLID) {
						int index = (tx + i) + (ty + j)*x + (tz + k)*x*y;
						T px = (tx+i) * dx + offset, py = (ty+j) * dx + offset, pz = (tz+k) * dx + offset, t;
						Vec3<T> curPosx(px + offset, py, pz), curPosy(px, py + offset, pz), curPosz(px, py, pz + offset);
						
						t = a.kWeight(p.p - curPosx);
						atomicAdd(&(sumkx[index]), t);
						atomicAdd(&(sumukx[index]), p.v.x*t);

						t = a.kWeight(p.p - curPosy);
						atomicAdd(&(sumky[index]), t);
						atomicAdd(&(sumuky[index]), p.v.y*t);

						t = a.kWeight(p.p - curPosz);
						atomicAdd(&(sumkz[index]), t);
						atomicAdd(&(sumukz[index]), p.v.z*t);
					}
				}
			}
		}
	}
	__syncthreads();
	for (int i = threadIdx.x; i < a.size; i += blockDim.x) {
		atomicAdd(&(a.a[i].sumuk.x), sumukx[i]);
		atomicAdd(&(a.a[i].sumuk.y), sumuky[i]);
		atomicAdd(&(a.a[i].sumuk.z), sumukz[i]);

		atomicAdd(&(a.a[i].sumk.x), sumkx[i]);
		atomicAdd(&(a.a[i].sumk.y), sumky[i]);
		atomicAdd(&(a.a[i].sumk.z), sumkz[i]);
	}
}

template <typename T>
__device__ void CuGrid<T>::interpU() {
	int o = gridDim.x*blockDim.x;
	T offset = dx / 2;
	for (int l = threadIdx.x + blockIdx.x*blockDim.x; l < numParticles; l += o) {
		Particle<T> p = this->list[l];
		int tx = floor(p.p.x / dx), ty = floor(p.p.y / dx), tz = floor(p.p.z / dx);
		for (int i = -2; i < 3; ++i) {
			for (int j = -2; j < 3; ++j) {
				for (int k = -2; k < 3; ++k) {
					if (a.get(tx + i, ty + j, tz + k).t != SOLID) {
						Voxel<T>& v = a.get(tx+i, ty+j, tz+k);
						T px = (tx + i) * dx + offset, py = (ty + j) * dx + offset, pz = (tz + k) * dx + offset, t;
						Vec3<T> curPosx(px + offset, py, pz), curPosy(px, py + offset, pz), curPosz(px, py, pz + offset);

						t = a.kWeight(p.p - curPosx);
						atomicAdd(&(v.sumk.x), t);
						atomicAdd(&(v.sumuk.x), p.v.x*t);

						t = a.kWeight(p.p - curPosy);
						atomicAdd(&(v.sumk.y), t);
						atomicAdd(&(v.sumuk.y), p.v.y*t);

						t = a.kWeight(p.p - curPosz);
						atomicAdd(&(v.sumk.z), t);
						atomicAdd(&(v.sumuk.z), p.v.z*t);
					}
				}
			}
		}
	}
}

template <typename T>
void CuGrid<T>::interpUCPU() {
	for (int p = 0; p < numParticles; ++p) {
		int i = floor(this->list[p].p.x / dx), j = floor(this->list[p].p.y / dx), k = floor(this->list[p].p.z / dx);
		T t;
		for (int ioff = -2; ioff < 3; ++ioff) {
			for (int joff = -2; joff < 3; ++joff) {
				for (int koff = -2; koff < 3; ++koff) {
					t = a.kWeight(this->list[p].p - Vec3<T>((i + ioff + 1.0)*dx, (j + joff + 0.5)*dx, (k + koff + 0.5)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.x += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.x += this->list[p].v.x*t;

					t = a.kWeight(this->list[p].p - Vec3<T>((i + ioff + 0.5)*dx, (j + joff + 1.0)*dx, (k + koff + 0.5)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.y += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.y += this->list[p].v.y*t;

					t = a.kWeight(this->list[p].p - Vec3<T>((i + ioff + 0.5)*dx, (j + joff + 0.5)*dx, (k + koff + 1.0)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.z += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.z += this->list[p].v.z*t;
				}
			}
		}
	}
}

template <typename T>
void CuGrid<T>::maxUCPU() {
	T max = 0;
	for (int p = 0; p < numParticles; ++p) {
		if (abs(this->list[p].v.x) > max)
			max = abs(this->list[p].v.x);
		if (abs(this->list[p].v.y) > max)
			max = abs(this->list[p].v.y);
		if (abs(this->list[p].v.z) > max)
			max = abs(this->list[p].v.z);
	}
	a.mU = max;
}

template <typename T>
void CuGrid<T>::setSubSamples(int n) {
	if(n > 0)
		subSamples = n;
}