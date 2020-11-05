#pragma once
#include "Vec3.h"

namespace Cu {
	template <typename T>
	class Triangle {
	public:
		Vec3<T> v[3];
	};

#include "Triangle.hpp"
}