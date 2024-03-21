#pragma once
#include "Transform.h"
#include <string>

namespace pathtracex {
	class Model {
	public:
		Transform transform{};

		std::string name = "Model";
	};
}