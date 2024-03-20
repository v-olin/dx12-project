#pragma once
#include <DirectXMath.h>

namespace pathtracex {
	class Transform
	{
	public:
		DirectX::XMMATRIX transformMatrix{};

		// TODO: Add rotation

		DirectX::XMFLOAT3 getPosition() const;
		DirectX::XMFLOAT3 getScale() const;

		void setPosition(DirectX::XMFLOAT3 position);
		void setScale(const DirectX::XMFLOAT3& scale);

		void translate(const DirectX::XMFLOAT3& translation);
		void scale(const DirectX::XMFLOAT3& scale);
	};
}