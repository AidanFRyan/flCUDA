#include "Voxel.h"

template <typename T>
Voxel<T>::Voxel(){
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
Voxel<T>::Voxel(bool in) : Voxel(){
	invalid = in;
	if(invalid)
		t = SOLID;
	// f.z = -9.8f;
	// numParticles = 0;
	// t = EMPTY;
}

template <typename T>
Voxel<T>::Voxel(const Voxel<T>& in) : Voxel(){
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
const Voxel<T>& Voxel<T>::operator=(const Voxel<T>& in){
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
void Voxel<T>::applyBodyForces(T dt){
	u.z += dt*-9.8;						//add dt*f to voxel velocity
}