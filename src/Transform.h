#pragma once
#include "Helper.h"

namespace pathtracex {
	class Transform {
	public:
		Transform();

		DirectX::XMMATRIX transformMatrix = DirectX::XMMatrixIdentity();
		DirectX::XMMATRIX rotationMatrix = DirectX::XMMatrixIdentity();

		float3 getPosition() const;
		float3 getScale() const;
		float3 getForward() const;
		float3 getUp() const;
		float3 getRight() const;

		DirectX::XMMATRIX getModelMatrix() const;

		void setPosition(float3 position);
		void setScale(const float3& scale);

		void translate(const float3& translation);
		void scale(const float3& scale);
		void rotate(const float3& axis, const float angle);

	private:
		float3 forward, up, right;
	};
}