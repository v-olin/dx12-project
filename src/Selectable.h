#pragma once
#include <string>
#include "Serializable.h"

namespace pathtracex {
	class Selectable : public Serializable {
	public:
		virtual std::string getName() = 0;
	};
}