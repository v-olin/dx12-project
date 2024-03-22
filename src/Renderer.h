#pragma once
#include "Scene.h"
#include "RendererSettings.h"

namespace pathtracex {

	class DXRenderer {
	public:
		void renderScene(Scene& scene, RendererSettings& rendererSettings);

	};
}