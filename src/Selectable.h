#pragma once
#include <string>

namespace pathtracex {
	class Selectable {
	public:
		virtual std::string getName() = 0;
	};
}