#pragma once
#include "Transform.h"
#include "Selectable.h"

namespace pathtracex {
	class Light : public Selectable {
	public:
		Transform transform{};

		std::string name = "Light";

		std::string getName() override { return name;  };
	};
}