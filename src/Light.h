#pragma once
#include "Transform.h"
#include "Selectable.h"

namespace pathtracex {
	class Light : public Selectable {
	public:
		Transform transform{};

		std::string name = "Light";
		float3 color{ 1.f, 1.f, 1.f };
		float intensity{ 10.f };

		std::string getName() override { return name; };

		std::vector<SerializableVariable> getSerializableVariables() override
		{
			return
			{
				{SerializableType::STRING, "Name", "The name of the light", &name},
				{SerializableType::MATRIX4X4, "TransformMatrix", "The transform matrix of the light", &transform.transformMatrix},
				{SerializableType::COLOR, "Color", "Light color", &color},
				{SerializableType::FLOAT, "Intensity", "Light intensity", &intensity}
			};
		};
	};
}