#pragma once
#include <vector>
#include <memory>
#include "Model.h"
#include "Camera.h"
#include "RenderSettings.h"
#include "Light.h"

namespace pathtracex {
	class Scene {
	public:
		std::vector<std::shared_ptr<Model>> models;
		std::vector<std::shared_ptr<Light>> lights;

		std::vector<std::shared_ptr<Model>> proceduralModels;

		std::string sceneName = "Scene";

		Camera camera{};
	};
}