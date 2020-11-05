#include "Vec3.h"
// using namespace Cu;

template <typename T>
Vec3<T>::Vec3(){
	x = 0;
	y = 0;
	z = 0;
}

template <typename T>
Vec3<T>::Vec3(T x, T y, T z){
	this->x = x;
	this->y = y;
	this->z = z;
}

template <typename T>
Vec3<T>::Vec3(const Vec3<T>& in){
	x = in.x;
	y = in.y;
	z = in.z;
}

template <typename T>
const Vec3<T>& Vec3<T>::operator=(const Vec3<T>& in){
	x = in.x;
	y = in.y;
	z = in.z;
	return *this;
}

template <typename T>
 Vec3<T> operator+(const Vec3<T>& in1, const Vec3<T>& in2){
	return Vec3<T>(in1.x+in2.x, in1.y+in2.y, in1.z+in2.z);
}

template <typename T>
 Vec3<T> operator-(const Vec3<T>& in1, const Vec3<T>& in2){
 	return Vec3<T>(in1.x-in2.x, in1.y-in2.y, in1.z-in2.z);
}

template <typename T>
const Vec3<T>& Vec3<T>::operator+=(const Vec3<T>& in){
	x += in.x;
	y += in.y;
	z += in.z;
	return *this;
}

template <typename T>
Vec3<T> operator*(const float& in1, const Vec3<T>& in2){
	return Vec3<T>(in1*in2.x, in1*in2.y, in1*in2.z);
}

template <typename T>
 Vec3<T> operator*(const Vec3<T>& in1, const float& in2){
	return in2*in1;
}

template <typename T>
 Vec3<T> operator/(const Vec3<T>& in1, const float& in2){
	return Vec3<T>(in1.x/in2, in1.y/in2, in1.z/in2);
}

 template <typename T>
 const Vec3<T>& Vec3<T>::operator=(const Point3& in) {
	 this->x = in.x;
	 this->y = in.y;
	 this->z = in.z;
	 return *this;
 }