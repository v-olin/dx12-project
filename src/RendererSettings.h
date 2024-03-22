#pragma once 

namespace pathtracex {
	struct RendererSettings {
		int width;
		int height;
		bool useMultiSampling = true;
		bool useRayTracing = false;
		int rayBounces = 10;
	};	
}