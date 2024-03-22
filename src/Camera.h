#pragma once
#include "Transform.h"

namespace pathtracex {
	class Camera {
	public:
		Transform transform{};

		float fov = 50.0f;
		float nearPlane = 0.1f;
		float farPlane = 1000.0f;
	};
}