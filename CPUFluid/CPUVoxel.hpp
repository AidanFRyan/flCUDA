#include "CPUVoxel.h"

template <typename T>
CPUVoxel<T>::CPUVoxel(){
	t = EMPTY;
	invalid = false;
	//numParticles = 0;
	aDiag = 0;
	aX = 0;
	aY = 0;
	aZ = 0;
	// dotTemp = 0;
	p = 0;
	divU = 0;
	res = 0;
	//particles.reserve(24);
}

template <typename T>
CPUVoxel<T>::CPUVoxel(bool in) : CPUVoxel(){
	invalid = in;
	if(invalid)
		t = SOLID;
	// f.z = -9.8f;
	// numParticles = 0;
	// t = EMPTY;
}

template <typename T>
CPUVoxel<T>::CPUVoxel(const CPUVoxel<T>& in) : CPUVoxel(){
	u = in.u;
	p = in.p;
	// t = EMPTY;
	// invalid = false;
	t = in.t;
	//numParticles = in.numParticles;
	//for(int i = 0; i < in.numParticles; i++){
	//	// particles[i] = in.particles[i];
	//	particles.push_back(in.particles[i]);
	//}
}

template <typename T>
const CPUVoxel<T>& CPUVoxel<T>::operator=(const CPUVoxel<T>& in){
	u = in.u;
	p = in.p;
	//numParticles = in.numParticles;
	invalid = in.invalid;
	t = in.t;
	//for(int i = 0; i < in.numParticles; i++){
	//	// particles[i] = in.particles[i];
	//	particles.push_back(in.particles[i]);
	//}
	return *this;
}

template <typename T>
void CPUVoxel<T>::applyBodyForces(T dt){
	u.z += dt*-9.8;						//add dt*f to CPUVoxel velocity
}