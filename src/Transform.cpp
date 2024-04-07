#include "Transform.h"

namespace dx = DirectX;

namespace pathtracex {

	Transform::Transform() :
		forward(1.f, 0.f, 0.f),
		up(0.f, 1.f, 0.f),
		right(0.f, 0.f, 1.0f)
	{ }

	float3 Transform::getPosition() const {
		return float3(transformMatrix.r[3].m128_f32);
	}

	float3 Transform::getScale() const
	{
		return float3(transformMatrix.r[0].m128_f32[0], transformMatrix.r[1].m128_f32[1], transformMatrix.r[2].m128_f32[2]);
	}

	float3 Transform::getForward() const {
		return forward;
	}

	float3 Transform::getUp() const {
		return up;
	}

	float3 Transform::getRight() const {
		return right;
	}

	dx::XMMATRIX Transform::getModelMatrix() const {
		return dx::XMMatrixMultiply(transformMatrix, rotationMatrix);
	}

	void Transform::setPosition(float3 position)
	{
		transformMatrix.r[3].m128_f32[0] = position.x;
		transformMatrix.r[3].m128_f32[1] = position.y;
		transformMatrix.r[3].m128_f32[2] = position.z;
	}

	void Transform::setScale(const float3& scale)
	{
		transformMatrix.r[0].m128_f32[0] = scale.x;
		transformMatrix.r[1].m128_f32[1] = scale.y;
		transformMatrix.r[2].m128_f32[2] = scale.z;
	}

	void Transform::translate(const float3& translation)
	{
		transformMatrix.r[3].m128_f32[0] += translation.x;
		transformMatrix.r[3].m128_f32[1] += translation.y;
		transformMatrix.r[3].m128_f32[2] += translation.z;
	}

	void Transform::scale(const float3& scale)
	{
		transformMatrix.r[0].m128_f32[0] *= scale.x;
		transformMatrix.r[1].m128_f32[1] *= scale.y;
		transformMatrix.r[2].m128_f32[2] *= scale.z;
	}

	void Transform::rotate(const float3& axis, const float angle) {
		dx::XMVECTOR vaxis = dx::XMLoadFloat3(&axis);
		dx::XMMATRIX newRotation = dx::XMMatrixRotationAxis(vaxis, angle);

		rotationMatrix = dx::XMMatrixMultiply(rotationMatrix, newRotation);

		forward = dx::XMVector3Normalize(dx::XMVector3Transform(float3{ 1.f, 0.f, 0.f }, rotationMatrix));
		up = dx::XMVector3Normalize(dx::XMVector3Transform(float3{ 0.f, 1.f, 0.f }, rotationMatrix));
		right = dx::XMVector3Normalize(dx::XMVector3Transform(float3{ 0.f, 0.f, 1.f }, rotationMatrix));
	}
}
