#pragma once 

#include "Camera.h"

namespace pathtracex {
	struct RenderSettings {
		int width;
		int height;
		bool useMultiSampling = true;
		bool useRayTracing = false;
		int rayBounces = 10;
		Camera& camera;

		RenderSettings(int width, int height, Camera& camera) :
			width(width), height(height),
			useMultiSampling(true),
			useRayTracing(false),
			rayBounces(10),
			camera(camera) { }

		void operator=(const RenderSettings& rs) {
			this->width = rs.width;
			this->height = rs.height;
			this->useMultiSampling = rs.useMultiSampling;
			this->useRayTracing = rs.useRayTracing;
			this->rayBounces = rs.rayBounces;
			this->camera = rs.camera;
		}
	};	
}