#pragma once
#include <vector>
#include <memory>
#include "Model.h"
#include "Camera.h"
#include "RendererSettings.h"

namespace pathtracex {
	class Scene {
	public:
		std::vector<std::shared_ptr<Model>> models;

		RendererSettings rendererSettings{};

		Camera camera{};
	};
}