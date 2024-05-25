#pragma once 

#include "Camera.h"

namespace pathtracex {
	struct RenderSettings {
		int width;
		int height;
		bool useMultiSampling = true;
		bool drawBoundingBox = false;
		bool useFrustumCulling = false;
		bool useBloomingEffect = false;
		bool drawProcedualWorld = false;
		bool useRayTracing = false;
		bool raytracingSupported = false;
		bool useTAA = false;
		bool useVSYNC = false;
		int rayBounces = 10;
		Camera& camera;

		RenderSettings(int width, int height, Camera& camera) :
			width(width), height(height),
			useMultiSampling(true),
			drawBoundingBox(false),
			useFrustumCulling(false),
			useBloomingEffect(false),
			useRayTracing(false),
			rayBounces(10),
			camera(camera) { }

		void operator=(const RenderSettings& rs) {
			this->width = rs.width;
			this->height = rs.height;
			this->useMultiSampling = rs.useMultiSampling;
			this->drawBoundingBox = rs.drawBoundingBox;
			this->useFrustumCulling = rs.useFrustumCulling;
			this->useBloomingEffect = rs.useBloomingEffect;
			this->useRayTracing = rs.useRayTracing;
			this->rayBounces = rs.rayBounces;
			this->camera = rs.camera;
			this->drawProcedualWorld = rs.drawProcedualWorld;
		}
	};	
}