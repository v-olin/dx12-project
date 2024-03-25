#pragma once
#include "Scene.h"
#include "RenderSettings.h"

namespace pathtracex {

	class DXRenderer {
	public:
		void renderScene(Scene& scene, RenderSettings& rendererSettings);

	};
}