#pragma once
#include <DirectXMath.h>
namespace pathtracex {
	struct Vertex {
		Vertex(float x, float y, float z, float r, float g, float b, float a) : pos(x, y, z), color(r, g, b, z) {}
		Vertex(DirectX::XMFLOAT3 pos, DirectX::XMFLOAT4 color, DirectX::XMFLOAT3 normal, DirectX::XMFLOAT2 tex) : pos(pos), color(color), tex(tex), normal(normal) {}
		Vertex(){}
		DirectX::XMFLOAT3 pos{};
		DirectX::XMFLOAT4 color{};
		DirectX::XMFLOAT2 tex{};
		//unsigned int has_col_tex{ 0 };
		DirectX::XMFLOAT3 normal{};
	};
}