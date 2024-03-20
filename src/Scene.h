#pragma once
#include <vector>
#include <memory>
#include "Model.h"

namespace pathtracex {
	class Scene {
		std::vector<std::shared_ptr<Model>> models;
	};
}