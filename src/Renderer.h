#pragma once
#include "Scene.h"

namespace pathtracex {
	struct RendererSettings {
		int width;
		int height;
		bool useMultiSampling;
	};

	class Renderer {
	public:
		void renderScene(Scene& scene, RendererSettings& rendererSettings);

	};
}