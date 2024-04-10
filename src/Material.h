#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <memory>

#include "Helper.h"
#include "Texture.h"

namespace pathtracex {
	class Material {
	public:
		std::string name;
		float3 color;
		float shininess;
		float metalness;
		float fresnel;
		float3 emission;
		float transparency;
		float ior;
		Texture colorTexture;
		Texture shininessTexture;
		Texture metalnessTexture;
		Texture fresnelTexture;
		Texture emissionTexture;
		Texture normalTexture;
	};
}