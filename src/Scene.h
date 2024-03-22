#pragma once
#include <vector>
#include <memory>
#include "Model.h"
#include "Camera.h"
#include "RendererSettings.h"
#include "Light.h"

namespace pathtracex {
	class Scene {
	public:
		std::vector<std::shared_ptr<Model>> models;
		std::vector<std::shared_ptr<Light>> lights;

		RendererSettings rendererSettings{};

		Camera camera{};
	};
}