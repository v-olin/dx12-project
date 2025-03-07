#pragma once
#include "Helper.h"
#include <DirectXMath.h>
namespace pathtracex {
	struct Vertex {
		Vertex(float x, float y, float z, float r, float g, float b, float a) : pos(x, y, z), color(r, g, b, z) {}
		Vertex(DirectX::XMFLOAT3 pos, DirectX::XMFLOAT4 color, DirectX::XMFLOAT3 normal, DirectX::XMFLOAT2 tex) : pos(pos), color(color), tex(tex), normal(normal) {}
		Vertex(float3 pos) : pos(pos) {};
		Vertex(){}
		DirectX::XMFLOAT3 pos{};
		DirectX::XMFLOAT4 color{};
		DirectX::XMFLOAT3 normal{};
		DirectX::XMFLOAT2 tex{};
		DirectX::XMFLOAT3 tangent{};
		unsigned int materialIdx;
	};
}