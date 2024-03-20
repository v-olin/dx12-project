#pragma once
#include "Scene.h"

namespace pathtracex {
	struct RendererSettings {
		int width;
		int height;
		bool useMultiSampling;
	};

	class DXRenderer {
	public:
		void renderScene(Scene& scene, RendererSettings& rendererSettings);

	};
}