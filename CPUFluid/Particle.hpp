#include "Particle.h"
// using namespace Cu;

template <typename T>
const Particle<T>& Particle<T>::operator=(const Particle<T>& in){
	v = in.v;
	v = in.vOld;
	p = in.p;
	// printf("in %f %f %f this %f %f %f\n", in.v.x, in.v.y, in.v.z, v.x, v.y, v.z);
	return *this;
}