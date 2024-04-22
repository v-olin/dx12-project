#pragma once
#include <DirectXMath.h>


namespace pathtracex {
	class Culling {
	public:
		// Singleton pattern to ensure only one instance of the class is created
		static Culling& getInstance() {
			static Culling instance;
			return instance;
		}

		// Delete copy and move constructors and assign operators
		Culling(const Culling&) = delete;
		Culling& operator=(const Culling&) = delete;

		// Update the frustum planes based on the view and projection matrices
		static void updateFrustum(const DirectX::XMMATRIX& view, const DirectX::XMMATRIX& projection);
		static bool isAABBInFrustum(const DirectX::XMFLOAT3& min, const DirectX::XMFLOAT3& max);
	private:
		Culling() {}

		static DirectX::XMFLOAT4 frustumPlanes[6];
	};
}