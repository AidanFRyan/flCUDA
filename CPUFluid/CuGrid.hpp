#include "CuGrid.h"

using namespace Cu;

template <typename T>
CuGrid<T>::CuGrid() {
	x = 0;
	y = 0;
	z = 0;
}

template <typename T>
CuGrid<T>::CuGrid(int x, int y, int z, double dx, double dt, double density) {
	this->x = x;
	this->y = y;
	this->z = z;
	printf("allocing %lu bytes\n", sizeof(Voxel<T>)*x*y*z);
	this->a = std::move(TriVec<T>(x, y, z, dx, dt, density));

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
void CuGrid<T>::nullifyHostTriVecs(){
	a.a = nullptr;
}

template <typename T>
void CuGrid<T>::nullifyDeviceTriVecs(){
}

template <typename T>
void CuGrid<T>::printParts(string filename){
	ofstream f;
	f.open(filename);
	printf("%d particles\n", particles.size());
	for (int p = 0; p < particles.size(); p++) {
		f << particles[p].p.x << ' ' << particles[p].p.y << ' ' << particles[p].p.z << endl;
	}
	f.close();
}

template <typename T>
CuGrid<T>::CuGrid(const CuGrid<T>& in){
	this->x = in.x;
	this->y = in.y;
	this->z = in.z;
	this->dx = in.dx;
	this->dt = in.dt;
	this->density = in.density;

	this->a = std::move(TriVec<T>(x, y, z, dx, dt, density));
}

template <typename T>
const CuGrid<T>& CuGrid<T>::operator=(const CuGrid<T>& in){
	a = in.a;
	x = in.x;	
	y = in.y;
	z = in.z;
	dx = in.dx;
	dt = in.dt;
	density = in.density;
	return *this;
}

template <typename T>
CuGrid<T>::~CuGrid() {}

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
void CuGrid<T>::initializeParticles(){
	for(int i = 0; i < x*y*z; i++){
		if(a.a[i].t == FLUID){
			int tz = i/(x*y), ty = (i%(x*y))/x, tx = (i%(x*y))%x;
			for(int j = 0; j < 8; j++){
				Particle<T> p;
				p.p.x = dx * tx + dx * a.dist(a.gen);
				p.p.y = dx * ty + dx * a.dist(a.gen);
				p.p.z = dx * tz + dx * a.dist(a.gen);
				this->particles.push_back(p);
			} 
		}
	}
}

template <typename T>
void CuGrid<T>::applyU() {
#pragma omp parallel for
	for (int p = 0; p < particles.size(); p++) {
		a.interpUtoP(particles[p]);
	}
}

template <typename T>
void CuGrid<T>::interpU() {	//interpolate from particle velocities to grid
	
	for (int p = 0; p < particles.size(); ++p) {
		int i = floor(particles[p].p.x / dx), j = floor(particles[p].p.y / dx), k = floor(particles[p].p.z / dx);
		T t;
		for (int ioff = -1; ioff < 2; ++ioff) {
			for (int joff = -1; joff < 2; ++joff) {
				for (int koff = -1; koff < 2; ++koff) {
					t = a.kWeight(particles[p].p - Vec3<T>((i + ioff + 1)*dx, (j + joff + 0.5)*dx, (k + koff + 0.5)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.x += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.x += particles[p].v.x*t;

					t = a.kWeight(particles[p].p - Vec3<T>((i + ioff + 0.5)*dx, (j + joff + 1)*dx, (k + koff + 0.5)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.y += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.y += particles[p].v.y*t;

					t = a.kWeight(particles[p].p - Vec3<T>((i + ioff + 0.5)*dx, (j + joff + 0.5)*dx, (k + koff + 1)*dx));
					a.get(i + ioff, j + joff, k + koff).sumk.z += t;
					a.get(i + ioff, j + joff, k + koff).sumuk.z += particles[p].v.z*t;
				}
			}
		}
	}
}

template <typename T>
void CuGrid<T>::advectParticles(){
#pragma omp parallel for
	for (int p = 0; p < particles.size(); ++p) {
		Particle<T> k2 = particles[p], k3;
		k2.p += k2.v*(dt / 2.0);
		a.interpUtoP(k2);
		k3 = k2;
		k3.p += k3.v*dt*(3.0 / 4.0);
		a.interpUtoP(k3);
		particles[p].p += particles[p].v*dt*(2.0 / 9.0) + k2.v*dt*(3.0 / 9.0) + k3.v*dt*(4.0 / 9.0);
	}	
}

template <typename T>
void CuGrid<T>::maxU() {
	T max = 0;
	for (int p = 0; p < particles.size(); ++p) {
		if (abs(particles[p].v.x) > max)
			max = abs(particles[p].v.x);
		if (abs(particles[p].v.y) > max)
			max = abs(particles[p].v.y);
		if (abs(particles[p].v.z) > max)
			max = abs(particles[p].v.z);
	}
}

template <typename T>
void CuGrid<T>::setFluidCells() {
	for (int p = 0; p < particles.size(); ++p) {
		int i = floor(particles[p].p.x / dx), j = floor(particles[p].p.y / dx), k = floor(particles[p].p.z / dx);
		if (a.get(i, j, k).t == EMPTY) {
			a.get(i, j, k).t = FLUID;
		}
	}
}

template <typename T>
void CuGrid<T>::reinsertOOBParticles() {
#pragma omp parallel for
	for (int p = 0; p < particles.size(); ++p) {
		int i = floor(particles[p].p.x / dx), j = floor(particles[p].p.y / dx), k = floor(particles[p].p.z / dx);
		if (a.get(i, j, k).t == SOLID) {
			a.reinsertToFluid(particles[p]);
		}
	}
}