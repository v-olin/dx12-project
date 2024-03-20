#include "Transform.h"
namespace dx = DirectX;

namespace pathtracex
{
	dx::XMFLOAT3 Transform::getPosition() const
	{
		return dx::XMFLOAT3(transformMatrix.r[3].m128_f32);
	}

	DirectX::XMFLOAT3 Transform::getScale() const
	{
		return dx::XMFLOAT3(transformMatrix.r[0].m128_f32[0], transformMatrix.r[1].m128_f32[1], transformMatrix.r[2].m128_f32[2]);
	}

	void Transform::setPosition(dx::XMFLOAT3 position)
	{
		transformMatrix.r[3].m128_f32[0] = position.x;
		transformMatrix.r[3].m128_f32[1] = position.y;
		transformMatrix.r[3].m128_f32[2] = position.z;
	}
	void Transform::setScale(const DirectX::XMFLOAT3& scale)
	{
		transformMatrix.r[0].m128_f32[0] = scale.x;
		transformMatrix.r[1].m128_f32[1] = scale.y;
		transformMatrix.r[2].m128_f32[2] = scale.z;
	}
	void Transform::translate(const DirectX::XMFLOAT3& translation)
	{
		transformMatrix.r[3].m128_f32[0] += translation.x;
		transformMatrix.r[3].m128_f32[1] += translation.y;
		transformMatrix.r[3].m128_f32[2] += translation.z;
	}
	void Transform::scale(const DirectX::XMFLOAT3& scale)
	{
		transformMatrix.r[0].m128_f32[0] *= scale.x;
		transformMatrix.r[1].m128_f32[1] *= scale.y;
		transformMatrix.r[2].m128_f32[2] *= scale.z;
	}
}
