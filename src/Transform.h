#pragma once
#include "Helper.h"

namespace pathtracex {
	class Transform
	{
	public:
		DirectX::XMMATRIX transformMatrix{};

		// TODO: Add rotation

		float3 getPosition() const;
		float3 getScale() const;

		void setPosition(float3 position);
		void setScale(const float3& scale);

		void translate(const float3& translation);
		void scale(const float3& scale);
	};
}