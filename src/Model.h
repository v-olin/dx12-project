#pragma once
#include "Transform.h"
#include <string>
#include "Selectable.h"

namespace pathtracex {
	class Model : public Selectable {
	public:
		Transform transform{};

		std::string name = "Model";

		std::string getName() override { return name; }
	};
}