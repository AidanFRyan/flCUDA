#ifndef PARTICLE_H
#define PARTICLE_H
#include "Vec3.h"
using namespace Cu;
namespace Cu{
    template <typename T>
	class Particle{
		public:
		const Particle& operator=(const Particle& in);
		Vec3<T> v, p , vOld;
	};
    #include "Particle.hpp"
}
#endif