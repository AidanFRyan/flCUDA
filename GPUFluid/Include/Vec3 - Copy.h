#ifndef VEC3_H
#define VEC3_H
#include "Macros.h"
#include "Common.h"
#include <Thinking.h>
class FlipFluidBasisThreadData;
namespace Cu{
    template <typename T>
	class Vec3{
	public:
		__device__ __host__ Vec3();
		__device__ __host__ Vec3(T x, T y, T z);
		__device__ __host__ Vec3(const Vec3& in);
		__device__ __host__ const Vec3& operator=(const Vec3& in);
		__device__ __host__ const Vec3& operator+=(const Vec3& in);
		const Vec3& operator=(const Point3& in);
		T x, y, z;
	};

    template <typename T>
	__device__ __host__ Vec3<T> operator+(const Vec3<T>& in1, const Vec3<T>& in2);
	template <typename T>
	__device__ __host__ Vec3<T> operator-(const Vec3<T>& in1, const Vec3<T>& in2);
	template <typename T>
	__device__ __host__ Vec3<T> operator*(const float& in1, const Vec3<T>& in2);
	template <typename T>
	__device__ __host__ Vec3<T> operator*(const Vec3<T>& in1, const float& in2);
	template <typename T>
	__device__ __host__ Vec3<T> operator/(const Vec3<T>& in1, const float& in2);
    
    #include "Vec3.hpp"
}

#endif
