#include "Voxel.h"

template <typename T>
Voxel<T>::Voxel(){
	t = EMPTY;
	invalid = false;
	aDiag = 0;
	aX = 0;
	aY = 0;
	aZ = 0;
	anX = 0;
	anY = 0;
	anZ = 0;
	p = 0;
	divU = 0;
	pold = 0;
	pureFluid = false;
}

template <typename T>
Voxel<T>::Voxel(bool in) : Voxel(){
	invalid = in;
	if(invalid){
		t = SOLID;
	}
}

template <typename T>
Voxel<T>::Voxel(const Voxel<T>& in) : Voxel(){
	divU = in.divU;
	p = in.p;
	pold = in.pold;
	t = in.t;

	aX = in.aX;
	aY = in.aY;
	aZ = in.aZ;
	anX = in.anX;
	anY = in.anY;
	anZ = in.anZ;
	aDiag = in.aDiag;
	invalid = in.invalid;
}

template <typename T>
const Voxel<T>& Voxel<T>::operator=(const Voxel<T>& in){
	divU = in.divU;
	p = in.p;
	pold = in.pold;
	aDiag = in.aDiag;
	aX = in.aX;
	aY = in.aY;
	aZ = in.aZ;
	anX = in.anX;
	anY = in.anY;
	anZ = in.anZ;
	invalid = in.invalid;
	t = in.t;
	return *this;
}