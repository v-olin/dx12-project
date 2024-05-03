#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <memory>

#include "Helper.h"
#include "Texture.h"

namespace pathtracex {
	class Material : public Serializable {
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
		Texture colorTexture2;
		Texture shininessTexture;
		Texture metalnessTexture;
		Texture fresnelTexture;
		Texture emissionTexture;
		Texture normalTexture;
		Texture normalTexture2;

		//Each material now has a descriptor heap that will contain all the textures
		ID3D12DescriptorHeap* mainDescriptorHeap;

		std::vector<SerializableVariable> getSerializableVariables() override
		{
			return
			{
				{SerializableType::STRING, "Name", "The name of the material", &name},
				{SerializableType::COLOR, "Color", "The color of the material", &color},
				{SerializableType::FLOAT, "Shininess", "The shininess of the material", &shininess},
				{SerializableType::FLOAT, "Metalness", "The metalness of the material", &metalness},
				{SerializableType::FLOAT, "Fresnel", "The fresnel of the material", &fresnel},
				{SerializableType::VECTOR3, "Emission", "The emission of the material", &emission},
				{SerializableType::FLOAT, "Transparency", "The transparency of the material", &transparency},
				{SerializableType::FLOAT, "IOR", "The index of refraction of the material", &ior},
			};
		};

		static Material createDefaultMaterial()
		{
			Material material;
			material.name = "Default Material";
			material.color = float3(1, 1, 1);
			material.shininess = 0.5f;
			material.metalness = 0.0f;
			material.fresnel = 0.0f;
			material.emission = float3(0, 0, 0);
			material.transparency = 0.0f;
			material.ior = 1.0f;
			material.colorTexture = Texture();
			material.shininessTexture = Texture();
			material.metalnessTexture = Texture();
			material.fresnelTexture = Texture();
			material.emissionTexture = Texture();
			material.normalTexture = Texture();
			return material;
		}
	};
}