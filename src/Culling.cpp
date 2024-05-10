#include "Culling.h"
#include "Logger.h"
#include "Model.h"
#include "Transform.h"

namespace pathtracex {
	// Initialize the frustum planes
	DirectX::XMFLOAT4 Culling::frustumPlanes[6];

	void Culling::updateFrustum(const DirectX::XMMATRIX& view, const DirectX::XMMATRIX& projection) {
		DirectX::XMMATRIX viewProjection = view * projection;

		DirectX::XMFLOAT4X4 vp;
		DirectX::XMStoreFloat4x4(&vp, viewProjection);

		// Left plane
		frustumPlanes[0].x = vp._14 + vp._11;
		frustumPlanes[0].y = vp._24 + vp._21;
		frustumPlanes[0].z = vp._34 + vp._31;
		frustumPlanes[0].w = vp._44 + vp._41;

		// Right plane
		frustumPlanes[1].x = vp._14 - vp._11;
		frustumPlanes[1].y = vp._24 - vp._21;
		frustumPlanes[1].z = vp._34 - vp._31;
		frustumPlanes[1].w = vp._44 - vp._41;

		// Top plane
		frustumPlanes[2].x = vp._14 - vp._12;
		frustumPlanes[2].y = vp._24 - vp._22;
		frustumPlanes[2].z = vp._34 - vp._32;
		frustumPlanes[2].w = vp._44 - vp._42;

		// Bottom plane
		frustumPlanes[3].x = vp._14 + vp._12;
		frustumPlanes[3].y = vp._24 + vp._22;
		frustumPlanes[3].z = vp._34 + vp._32;
		frustumPlanes[3].w = vp._44 + vp._42;

		// Near plane
		frustumPlanes[4].x = vp._13;
		frustumPlanes[4].y = vp._23;
		frustumPlanes[4].z = vp._33;
		frustumPlanes[4].w = vp._43;

		// Far plane
		frustumPlanes[5].x = vp._14 - vp._13;
		frustumPlanes[5].y = vp._24 - vp._23;
		frustumPlanes[5].z = vp._34 - vp._33;
		frustumPlanes[5].w = vp._44 - vp._43;

		// Normalize the planes
		for (int i = 0; i < 6; ++i)
		{
			float length = sqrt((frustumPlanes[i].x * frustumPlanes[i].x) + (frustumPlanes[i].y * frustumPlanes[i].y) + (frustumPlanes[i].z * frustumPlanes[i].z));
			frustumPlanes[i].x /= length;
			frustumPlanes[i].y /= length;
			frustumPlanes[i].z /= length;
			frustumPlanes[i].w /= length;
		}
	}

	// Check if a AABB is inside the frustum
	bool Culling::isAABBInFrustum(const DirectX::XMFLOAT3& min, const DirectX::XMFLOAT3& max, const DirectX::XMMATRIX& modelTransformMatrix) {
		for (DirectX::XMFLOAT4 plane : frustumPlanes) {
			// Calculate the normal and constant of the plane
			DirectX::XMVECTOR planeNormal = DirectX::XMVectorSet(plane.x, plane.y, plane.z, 0.0f);
			float planeConstant = plane.w;

			// Calculate the furthest point in the direction of the normal
			DirectX::XMFLOAT3 furthestPoint;

			// X-axis
			if (plane.x < 0) {
				furthestPoint.x = min.x;
			}
			else {
				furthestPoint.x = max.x;
			}

			// Y-axis
			if (plane.y < 0) {
				furthestPoint.y = min.y;
			}
			else {
				furthestPoint.y = max.y;
			}

			// Z-axis
			if (plane.z < 0) {
				furthestPoint.z = min.z;
			}
			else {
				furthestPoint.z = max.z;
			}

			// Transform the furthest point to world space
			DirectX::XMVECTOR transformedFurthestPoint = DirectX::XMVector3Transform(DirectX::XMLoadFloat3(&furthestPoint), modelTransformMatrix);

			// Calculate the distance from the plane
			float distance = DirectX::XMVectorGetX(DirectX::XMVector3Dot(planeNormal, transformedFurthestPoint)) + planeConstant;

			// If the distance is negative, the AABB is outside the frustum
			if (distance < 0) {
				return false;
			}
		}
		return true;
	}
}