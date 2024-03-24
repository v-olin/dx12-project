#pragma once
#include <DirectXMath.h>
namespace pathtracex {
	struct Vertex {
		Vertex(float x, float y, float z, float r, float g, float b, float a) : pos(x, y, z), color(r, g, b, z) {}
		DirectX::XMFLOAT3 pos;
		DirectX::XMFLOAT4 color;
	};
}