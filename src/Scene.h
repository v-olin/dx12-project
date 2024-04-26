#pragma once
#include <vector>
#include <memory>
#include "Model.h"
#include "Camera.h"
#include "RenderSettings.h"
#include "Light.h"
#include "ProcedualWorldManager.h"

namespace pathtracex {
	class Scene {
	public:
		std::vector<std::shared_ptr<Model>> models;
		std::vector<std::shared_ptr<Light>> lights;

		std::vector<std::shared_ptr<Model>> proceduralGroundModels;
		std::vector<std::shared_ptr<Model>> proceduralSkyModels;

		std::string sceneName = "Scene";

		//Camera camera{};
		ProcedualWorldManager* procedualWorldManager;
	};
}