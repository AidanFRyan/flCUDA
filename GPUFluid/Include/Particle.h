#ifndef PARTICLE_H
#define PARTICLE_H
#include "Vec3.h"
namespace Cu{
    template <typename T>
	class Particle{
		public:
			__device__ __host__ const Particle& operator=(const Particle& in);
			Vec3<T> v, p;
	};
    #include "Particle.hpp"
}
#endif