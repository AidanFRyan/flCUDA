#ifndef COMMON_H
#define COMMON_H
//#include <omp.h>
#include <ppl.h>
#include <Windows.h>

template <typename T>
__device__ __host__ T linterp(const T& a, const T& p0, const T& p1) {
	return (1 - a) * p0 + a * p1;
}

template <typename T>
__device__ __host__ T trilinterp(const T& aX, const T& aY, const T& aZ, const T& p000, const T& p001, const T& p010, const T& p011, const T& p100, const T& p101, const T& p110, const T& p111) {
	return linterp(aZ, linterp(aY, linterp(aX, p000, p001), linterp(aX, p010, p011)), linterp(aY, linterp(aX, p100, p101), linterp(aX, p110, p111)));
}

#endif